// ============================================================================
// yolo_dpu_infer.cpp  —  M4: VART C++ host code (pre-staging, highest weight)
// ----------------------------------------------------------------------------
// รันบน PS (Arm Cortex-A53) ของ KV260. โหลด .xmodel -> สร้าง DPU runner ->
// preprocess ภาพ -> รัน DPU -> ดึง raw output 3 tensors -> dequantize ->
// (ทางเลือก) dump เป็น .bin เพื่อเทียบกับ PyTorch golden ใน M5.
//
// เป้าหมาย gate ของไฟล์นี้ (M4): เขียน + compile-check ผ่านบน host.
// รันจริงบนบอร์ด = Phase 1 (M9) เมื่อบอร์ดมาถึง.
//
// --- I/O spec (จาก Phase 0 log; ยืนยันซ้ำด้วย `xdputil xmodel <f>.xmodel -l`) ---
//   input : [1, 640, 640, 3]  NHWC  int8   (fixpos อ่านจาก tensor ตอน runtime)
//   output: 3 tensors int8, NHWC, แยกกันด้วยขนาด spatial 80x80 / 40x40 / 20x20
//           channel ขึ้นกับหัวโมเดล:
//             - YOLOv8n (Track A, artifact ที่ผ่าน gate ตอนนี้): 144 = 64 DFL + 80 class
//             - YOLO26n (Track B, โมเดลเป้าหมาย): 84 = 4 box + 80 class (reg_max=1, ไม่มี DFL)
//           โค้ดนี้ทำแค่ดึง raw + dequant จึงรับได้ทั้งสองหัว (decode ต่างกัน = งาน PS ขั้นถัดไป)
//   DPU   : DPUCZDX8G_ISA1_B4096
//
// --- จุดเสี่ยง (เขียนกันไว้ตาม PreStaging_Plan) ---
//   [1] NHWC vs NCHW : VART รับ NHWC. preprocessing (cv2) ให้ HWC อยู่แล้ว —
//       *อย่า* transpose เป็น CHW ก่อนป้อน DPU (นั่นสำหรับ PyTorch path เท่านั้น).
//   [2] fixpos/scale : input int8 = round(float * 2^fixpos); output float =
//       int8 * 2^(-fixpos). ต้อง "อ่าน fixpos จาก tensor" ไม่ hardcode —
//       ถ้า re-compile ด้วย arch อื่น fixpos เปลี่ยนได้.
//   [3] dequantize เป็นงาน PS : DPU คืน int8, ต้องแปลงกลับ float บน PS
//       ก่อนส่งไป decode (DFL softmax+conv, sigmoid) ในขั้นถัดไป.
//
// Build: ดู CMakeLists.txt / build.sh (compile-check ใน Vitis AI 3.0 container).
// ============================================================================

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

// ----------------------------------------------------------------------------
// หา DPU subgraph ตัวเดียว (Phase 0 gate = "DPU subgraph number 1").
// XIR แบ่ง graph เป็น subgraph หลายตัว (บาง subgraph = CPU/USER). เอาเฉพาะที่
// attr "device" == "DPU". ถ้าไม่เจอ หรือเจอมากกว่า 1 -> ผิด spec, abort.
// ----------------------------------------------------------------------------
static const xir::Subgraph* get_dpu_subgraph(const xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  const xir::Subgraph* dpu = nullptr;
  int dpu_count = 0;
  for (auto* c : children) {
    if (c->has_attr("device") &&
        c->get_attr<std::string>("device") == "DPU") {
      dpu = c;
      dpu_count++;
    }
  }
  CHECK_EQ(dpu_count, 1)
      << "expected exactly 1 DPU subgraph (Phase 0 gate), got " << dpu_count;
  return dpu;
}

// scale สำหรับ quantize/dequantize จาก fixpos ของ tensor.
// input : float -> int8  ใช้  2^(fix_point)
// output: int8  -> float ใช้  2^(-fix_point)
static float tensor_scale_pow2(const xir::Tensor* t) {
  CHECK(t->has_attr("fix_point"))
      << "tensor " << t->get_name() << " has no fix_point attr";
  int fixpos = t->get_attr<int>("fix_point");
  return std::exp2f(static_cast<float>(fixpos));
}

// ----------------------------------------------------------------------------
// Preprocess — ต้องตรงกับ calibration ของ Phase 0 เป๊ะ (load_calib_batch):
//   cv2.resize(im,(640,640))  [plain resize, ไม่ letterbox]
//   cv2.cvtColor(BGR2RGB)
//   /255.0
//   *ไม่* transpose (คง HWC สำหรับ VART/NHWC)
// เขียนผลลง int8 buffer โดย quantize ด้วย input scale (2^fixpos).
// ----------------------------------------------------------------------------
static void preprocess_into(const std::string& img_path, int H, int W,
                            float in_scale, int8_t* dst) {
  cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
  CHECK(!bgr.empty()) << "cannot read image: " << img_path;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(W, H));  // plain resize (ตรงกับ calib)

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);  // BGR -> RGB

  // HWC, /255, quantize -> int8. layout ปลายทาง = NHWC ตาม tensor.
  const int channels = 3;
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = rgb.ptr<uint8_t>(y);
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < channels; ++c) {
        float v = static_cast<float>(row[x * channels + c]) / 255.0f;  // [0,1]
        int q = static_cast<int>(std::round(v * in_scale));            // fixpos
        q = std::max(-128, std::min(127, q));                          // clamp
        dst[(y * W + x) * channels + c] = static_cast<int8_t>(q);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <model.xmodel> <image> [out_prefix]\n"
              << "  dumps dequantized raw outputs to <out_prefix>_<HxW>.bin "
                 "(float32, NHWC) for M5 comparison.\n";
    return 1;
  }
  const std::string xmodel = argv[1];
  const std::string image = argv[2];
  const std::string out_prefix = (argc >= 4) ? argv[3] : "";

  google::InitGoogleLogging(argv[0]);

  // 1) โหลด xmodel + หา DPU subgraph -----------------------------------------
  auto graph = xir::Graph::deserialize(xmodel);
  auto* subgraph = get_dpu_subgraph(graph.get());
  LOG(INFO) << "DPU subgraph: " << subgraph->get_name();

  // 2) สร้าง runner (RunnerExt = จัดการ TensorBuffer/quant scale ให้) ---------
  // VAI 3.0: create_runner รับ (subgraph, xir::Attrs*) — ส่ง attrs เปล่าไป
  auto attrs = xir::Attrs::create();
  auto runner = vart::RunnerExt::create_runner(subgraph, attrs.get());
  auto input_tbs = runner->get_inputs();
  auto output_tbs = runner->get_outputs();
  CHECK_EQ(input_tbs.size(), 1u) << "expected exactly 1 input tensor";
  CHECK_EQ(output_tbs.size(), 3u) << "expected exactly 3 output tensors";

  // 3) อ่าน input spec -------------------------------------------------------
  auto* in_tensor = input_tbs[0]->get_tensor();
  auto in_shape = in_tensor->get_shape();  // {1,640,640,3} NHWC
  CHECK_EQ(in_shape.size(), 4u);
  const int N = in_shape[0], H = in_shape[1], W = in_shape[2], C = in_shape[3];
  const float in_scale = tensor_scale_pow2(in_tensor);
  LOG(INFO) << "input " << in_tensor->get_name() << "  shape=[" << N << ","
            << H << "," << W << "," << C << "]  fixpos_scale=" << in_scale;
  CHECK_EQ(N, 1) << "this host code assumes batch=1";
  CHECK_EQ(C, 3) << "expected 3-channel RGB input";

  // 4) preprocess -> เขียนลง input TensorBuffer ------------------------------
  {
    uint64_t addr = 0u;
    size_t size = 0u;
    std::tie(addr, size) = input_tbs[0]->data(std::vector<int32_t>{0, 0, 0, 0});
    CHECK_GE(size, static_cast<size_t>(H) * W * C);
    preprocess_into(image, H, W, in_scale, reinterpret_cast<int8_t*>(addr));
  }
  for (auto& tb : input_tbs) {
    tb->sync_for_write(0, tb->get_tensor()->get_data_size() /
                              tb->get_tensor()->get_shape()[0]);
  }

  // 5) รัน DPU (async + wait) ------------------------------------------------
  auto job = runner->execute_async(input_tbs, output_tbs);
  runner->wait(static_cast<int>(job.first), -1 /*forever*/);

  for (auto& tb : output_tbs) {
    tb->sync_for_read(0, tb->get_tensor()->get_data_size() /
                             tb->get_tensor()->get_shape()[0]);
  }

  // 6) ดึง raw outputs -> dequantize -> (ทางเลือก) dump .bin ------------------
  // ระบุแต่ละ tensor ด้วย "ขนาด spatial" ไม่ยึด index (order runtime ไม่การันตี).
  for (size_t i = 0; i < output_tbs.size(); ++i) {
    auto* ot = output_tbs[i]->get_tensor();
    auto os = ot->get_shape();  // {1, Hf, Wf, 144} NHWC
    CHECK_EQ(os.size(), 4u);
    const int Hf = os[1], Wf = os[2], Cf = os[3];
    const float out_scale = tensor_scale_pow2(ot);           // = 2^fixpos
    const float dequant = 1.0f / out_scale;                  // int8 -> float

    uint64_t addr = 0u;
    size_t size = 0u;
    std::tie(addr, size) =
        output_tbs[i]->data(std::vector<int32_t>{0, 0, 0, 0});
    const int8_t* src = reinterpret_cast<const int8_t*>(addr);
    const size_t n = static_cast<size_t>(Hf) * Wf * Cf;
    CHECK_GE(size, n);

    // สถิติเร็ว ๆ ไว้ sanity-check ว่า output ไม่ใช่ศูนย์ล้วน/อิ่มตัว
    float mn = 1e30f, mx = -1e30f;
    std::vector<float> deq(n);
    for (size_t k = 0; k < n; ++k) {
      float f = static_cast<float>(src[k]) * dequant;
      deq[k] = f;
      mn = std::min(mn, f);
      mx = std::max(mx, f);
    }
    LOG(INFO) << "output[" << i << "] " << ot->get_name() << "  shape=[1," << Hf
              << "," << Wf << "," << Cf << "]  fixpos_scale=" << out_scale
              << "  float_range=[" << mn << "," << mx << "]";
    // 144 = YOLOv8 head (64 DFL + 80 class); 84 = YOLO26 head (4 box + 80 class).
    // เตือนแต่ไม่ abort ถ้าเจอค่าอื่น (โมเดล/คลาสอาจต่างจากที่คาด).
    if (Cf != 144 && Cf != 84) {
      LOG(WARNING) << "  unexpected channel count " << Cf
                   << " (expected 144=v8 head or 84=yolo26 head)";
    }

    if (!out_prefix.empty()) {
      std::string fn = out_prefix + "_" + std::to_string(Hf) + "x" +
                       std::to_string(Wf) + ".bin";
      std::ofstream f(fn, std::ios::binary);
      f.write(reinterpret_cast<const char*>(deq.data()),
              static_cast<std::streamsize>(n * sizeof(float)));
      LOG(INFO) << "  wrote " << fn << " (float32 NHWC, " << n << " elems)";
    }
  }

  LOG(INFO) << "done. (M4 host path OK; decode DFL/sigmoid = next stage on PS)";
  return 0;
}
