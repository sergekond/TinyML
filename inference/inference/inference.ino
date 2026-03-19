#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "esp_camera.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#include "model_int8.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define MODEL_W 96
#define MODEL_H 96

// ===== Tensor Arena =====
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t *tensor_arena;

// ===== TFLite =====
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

float input_scale;
int input_zero;

int person_counter = 0;

unsigned long last_inference = 0;

// ================= PREPROCESS =================
void preprocess(camera_fb_t *fb, int8_t *input_data)
{
  int src_w = fb->width;
  int src_h = fb->height;

  int crop_x = (src_w - MODEL_W) / 2;
  int crop_y = (src_h - MODEL_H) / 2;

  uint8_t *buf = fb->buf;

  for(int y = 0; y < MODEL_H; y++)
  {
    for(int x = 0; x < MODEL_W; x++)
    {
      int sx = crop_x + x;
      int sy = crop_y + y;

      uint8_t gray = buf[sy * src_w + sx];

      float normalized = gray / 255.0f;

      int32_t q = normalized / input_scale + input_zero;

      if(q > 127) q = 127;
      if(q < -128) q = -128;

      input_data[y * MODEL_W + x] = (int8_t)q;
    }
  }
}

void inference () {
  camera_fb_t *fb = esp_camera_fb_get();

  if(!fb)
  {
    Serial.println("Camera capture failed");
    return;
  }

  preprocess(fb, input_tensor->data.int8);

  if(interpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("Invoke failed");
    esp_camera_fb_return(fb);
    return;
  }

  int8_t output = output_tensor->data.int8[0];

  float prob = (output - output_tensor->params.zero_point) *
               output_tensor->params.scale;

  Serial.print("Person probability: ");
  Serial.println(prob, 3);


  if(prob > 0.5)
    Serial.println("PERSON DETECTED");
  else
    Serial.println("OTHER");


  esp_camera_fb_return(fb);
}

// ================= SETUP =================
void setup()
{
  Serial.begin(115200);

  Serial.println("Starting TinyML ESP32-CAM");

  if(psramFound())
    Serial.println("PSRAM found");
  else
    Serial.println("PSRAM NOT FOUND");


  // ===== Camera =====
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  config.pixel_format = PIXFORMAT_GRAYSCALE;

  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);

  if(err != ESP_OK)
  {
    Serial.println("Camera init failed");
    return;
  }

  Serial.println("Camera OK");


  // ===== Tensor Arena =====
tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
if (!tensor_arena) {
    Serial.println("Failed to allocate internal RAM for tensor arena");
    return;
}

  if(!tensor_arena)
  {
    Serial.println("Tensor arena alloc failed");
    return;
  }


  // ===== Load Model =====
  tfl_model = tflite::GetModel(model);

  if(tfl_model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch");
    return;
  }


  // ===== Fast Ops Resolver =====
static tflite::MicroMutableOpResolver<7> resolver;

resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddRelu();
resolver.AddFullyConnected();
resolver.AddReshape();
resolver.AddLogistic();
resolver.AddMean();

  // ===== Interpreter =====
  static tflite::MicroInterpreter static_interpreter(
    tfl_model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter
  );

  interpreter = &static_interpreter;

  if(interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("AllocateTensors failed");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  input_scale = input_tensor->params.scale;
  input_zero = input_tensor->params.zero_point;

  Serial.println("TFLite initialized");

  Serial.print("Input scale: ");
  Serial.println(input_scale, 6);

  Serial.print("Input zero: ");
  Serial.println(input_zero);
}

void loop() {
  if (millis() - last_inference >= 500) {
    last_inference = millis();
      inference();
    }
}
