//
// Created by Line on 2021/4/9.
//
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <Interpreter.hpp>
#include <MNNDefine.h>
#include <Tensor.hpp>
#include <ImageProcess.hpp>
#include <MNNForwardType.h>
#include <fstream>
#include <functional>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../MNN/3rd_party/imageHelper/stb_image.h"

using namespace MNN;
using namespace MNN::CV;

std::shared_ptr<Interpreter> net_;
MNN::Session* session_ = NULL;
MNN::Tensor* input_ = nullptr;

extern "C" {
JNIEXPORT jstring JNICALL
Java_xindaface_XDFace_version(JNIEnv *env, jobject instance) {
    return env->NewStringUTF("version 1.0.0");
}

JNIEXPORT void JNICALL
Java_xindaface_XDFace_load(JNIEnv *env, jobject instance, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    AAsset* asset = AAssetManager_open(mgr, "mobilenet_v2_1.0_224.tflite.mnn", AASSET_MODE_UNKNOWN);
    off_t bufferSize = AAsset_getLength(asset);
    char *buffer=(char *)malloc(bufferSize);
    AAsset_read(asset, buffer, bufferSize);

    net_ = std::shared_ptr<MNN::Interpreter>(Interpreter::createFromBuffer(buffer, bufferSize));
    ScheduleConfig config;
//    config.type = MNN_FORWARD_CPU;
//    config.type = MNN_FORWARD_VULKAN;
    config.type = MNN_FORWARD_OPENCL;
//    config.type = MNN_FORWARD_OPENGL;
//    config.type = MNN_FORWARD_AUTO;
//    config.numThread = 4;
//    BackendConfig bnconfig;
//    bnconfig.precision = BackendConfig::Precision_Low;
//    bnconfig.power = MNN::BackendConfig::Power_High;
//    bnconfig.memory = MNN::BackendConfig::Memory_High;
//    config.backendConfig = &bnconfig;
    session_ = net_->createSession(config);
    input_ = net_->getSessionInput(session_, NULL);
    auto shape = input_->shape();
    shape[0] = 1;
    net_->resizeTensor(input_, shape);
    net_->resizeSession(session_);
}

JNIEXPORT jfloatArray JNICALL
Java_xindaface_XDFace_run(JNIEnv *env, jobject instance, jobject vBmp) {
    AndroidBitmapInfo  infocolor;
    void*              pixelscolor;
    AndroidBitmap_getInfo(env, vBmp, &infocolor);
    AndroidBitmap_lockPixels(env, vBmp, &pixelscolor);

    auto output = net_->getSessionOutput(session_, NULL);

    int size_h = input_->height();
    int size_w = input_->width();
    int width = infocolor.width;
    int height = infocolor.height;
    Matrix trans;
    trans.setScale((float) (width - 1) / (size_w - 1), (float) (height - 1) / (size_h - 1));
    ImageProcess::Config config;
    config.filterType = BILINEAR;
    float mean[3] = {103.94f, 116.78f, 123.68f};
    float normals[3] = {0.017f, 0.017f, 0.017f};
    // float mean[3]     = {127.5f, 127.5f, 127.5f};
    // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
    ::memcpy(config.mean, mean, sizeof(mean));
    ::memcpy(config.normal, normals, sizeof(normals));
    config.sourceFormat = RGBA;
    config.destFormat = BGR;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t *) pixelscolor, width, height, 0, input_);

    net_->runSession(session_);

    auto dimType = output->getDimensionType();
    std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));

    output->copyToHostTensor(outputUser.get());

    auto size = outputUser->elementSize();
    auto values = outputUser->host<float>();

    jfloatArray jRes = env->NewFloatArray(size);
    env->SetFloatArrayRegion(jRes, 0, size, values);
    return jRes;

}

JNIEXPORT jfloatArray JNICALL
Java_xindaface_XDFace_predict(JNIEnv *env, jobject instance, jobject assetManager, jobject vBmp) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    AAsset* asset = AAssetManager_open(mgr, "mobilenet_v2_1.0_224.tflite.mnn", AASSET_MODE_UNKNOWN);
    off_t bufferSize = AAsset_getLength(asset);
    char *buffer=(char *)malloc(bufferSize);
    AAsset_read(asset, buffer, bufferSize);

    AndroidBitmapInfo  infocolor;
    void*              pixelscolor;
    AndroidBitmap_getInfo(env, vBmp, &infocolor);
    AndroidBitmap_lockPixels(env, vBmp, &pixelscolor);

    std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer(buffer, bufferSize));
    ScheduleConfig config;
//    config.type = MNN_FORWARD_AUTO;
//    config.type = MNN_FORWARD_VULKAN;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0] = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto output = net->getSessionOutput(session, NULL);
    {
        int size_h = input->height();
        int size_w = input->width();
        int width = infocolor.width;
        int height = infocolor.height;
        Matrix trans;
        trans.setScale((float) (width - 1) / (size_w - 1), (float) (height - 1) / (size_h - 1));
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3] = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};
        // float mean[3]     = {127.5f, 127.5f, 127.5f};
        // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t *) pixelscolor, width, height, 0, input);
    }
    net->runSession(session);

    auto dimType = output->getDimensionType();
    std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));

    output->copyToHostTensor(outputUser.get());

    auto size = outputUser->elementSize();
    auto values = outputUser->host<float>();

    jfloatArray jRes = env->NewFloatArray(size);
    env->SetFloatArrayRegion(jRes, 0, size, values);
    return jRes;

}

JNIEXPORT jstring JNICALL
Java_xindaface_XDFace_pictureRecognition(JNIEnv *env, jobject instance) {
//    std::shared_ptr<Interpreter> net(Interpreter::createFromFile("/sdcard/pl/model/MobileNetV2_224.mnn"));
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile("/sdcard/pl/model/mobilenet_v2_1.0_224.tflite.mnn"));
    ScheduleConfig config;
//    config.type = MNN_FORWARD_AUTO;
    config.type = MNN_FORWARD_VULKAN;
    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0] = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto output = net->getSessionOutput(session, NULL);
    std::vector<std::string> words;

    std::ifstream inputOs("/sdcard/pl/model/synset_words.txt");
    std::string line;
    while (std::getline(inputOs, line)) {
        words.emplace_back(line);
    }
    {
        auto dims = input->shape();
        int inputDim = 0;
        int size_w = 0;
        int size_h = 0;
        int bpp = 0;
        bpp = input->channel();
        size_h = input->height();
        size_w = input->width();
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;

        int width, height, channel;
        auto inputImage = stbi_load("/sdcard/pl/model/testcat.jpg", &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            return 0;
        }
        Matrix trans;
        trans.setScale((float) (width - 1) / (size_w - 1), (float) (height - 1) / (size_h - 1));
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3] = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};
        // float mean[3]     = {127.5f, 127.5f, 127.5f};
        // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t *) inputImage, width, height, 0, input);
        stbi_image_free(inputImage);
    }
    net->runSession(session);
    std::string res = "";
    {
        auto dimType = output->getDimensionType();
        if (output->getType().code != halide_type_float) {
            dimType = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));

        output->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();

        auto size = outputUser->elementSize();
        std::vector<std::pair<int, float>> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_uint && type.bytes() == 1) {
            auto values = outputUser->host<uint8_t>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_int && type.bytes() == 1) {
            auto values = outputUser->host<int8_t>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        // Find Max
        std::sort(tempValues.begin(), tempValues.end(),
                  [](std::pair<int, float> a, std::pair<int, float> b) {
                      return a.second > b.second;
                  });

        int length = size > 10 ? 10 : size;
        if (words.empty()) {
            for (int i = 0; i < length; ++i) {
                res += std::to_string(tempValues[i].first) + " " + std::to_string(tempValues[i].second) + "\n";
            }
        } else {
            for (int i = 0; i < length; ++i) {
                res += words[tempValues[i].first] + " " + std::to_string(tempValues[i].second) + "\n";
            }
        }
    }
    return env->NewStringUTF(res.c_str());
}
}
