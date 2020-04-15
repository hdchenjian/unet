#include <string>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <unistd.h>
#include <thread>
#include <fstream>

#include "cnrt.h"

// caffe offline: 32FPS
bool LOG_ON = true;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

class ClassifierLauncher {
public:
    ClassifierLauncher(std::string offmodel);
    void run_network(cv::Mat &img1, std::vector<std::vector<float>> &detections);

    cnrtModel_t model_;
    cnrtFunction_t function_;
    cnrtRuntimeContext_t runtime_ctx_;

    cnrtQueue_t queue_;
    cnrtNotifier_t start_notifier_;
    cnrtNotifier_t end_notifier_;
    int64_t* inputSizeS_, *outputSizeS_;
    int inputNum_, outputNum_;
    void** inputMluPtrS;
    void** outputMluPtrS;
    int in_n_, in_c_, in_h_, in_w_;
    void* cpu_data_cast_type;
    float *cpu_data_;
    int *out_n_, *out_c_, *out_h_, *out_w_;
    void **output_data_cast;
    float **outputCpu;

    ~ClassifierLauncher() {
        cnrtFreeArray(inputMluPtrS, inputNum_);
        cnrtFreeArray(outputMluPtrS, outputNum_);

        cnrtDestroyFunction(function_);
        cnrtDestroyQueue(queue_);
        cnrtDestroyNotifier(&start_notifier_);
        cnrtDestroyNotifier(&end_notifier_);
        cnrtDestroyRuntimeContext(runtime_ctx_);
        cnrtUnloadModel(model_);

        free(cpu_data_cast_type);
        free(cpu_data_);
        free(out_n_);
        free(out_c_);
        free(out_h_);
        free(out_w_);
        for (int j = 0; j < outputNum_; j++){
            free(output_data_cast[j]);
            free(outputCpu[j]);
        }
        free(output_data_cast);
        free(outputCpu);
        //std::cout << "~ClassifierLauncher" << std::endl;
    }
};

ClassifierLauncher::ClassifierLauncher(std::string offmodel){
    int dev_id_ = 0;
    cnrtLoadModel(&model_, offmodel.c_str());
    std::string name = "subnet0";

    cnrtDev_t dev;
    CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_id_));
    CNRT_CHECK(cnrtSetCurrentDevice(dev));
    if (LOG_ON) {
        std::cout << "Init Classifier for device " << dev_id_ << std::endl;
    }

    CNRT_CHECK(cnrtCreateFunction(&(function_)));
    CNRT_CHECK(cnrtExtractFunction(&(function_), model_, name.c_str()));

    CNRT_CHECK(cnrtCreateRuntimeContext(&runtime_ctx_, function_, nullptr));
    cnrtSetRuntimeContextDeviceId(runtime_ctx_, dev_id_);

    if (cnrtInitRuntimeContext(runtime_ctx_, nullptr) != CNRT_RET_SUCCESS) {
        std::cout << "Failed to init runtime context" << std::endl;
        return;
    }

    CNRT_CHECK(cnrtGetInputDataSize(&inputSizeS_, &inputNum_, function_));
    if (LOG_ON) {
        std::cout << "model input num: " << inputNum_ << " input size " << *inputSizeS_ << std::endl;
    }
    CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeS_, &outputNum_, function_));
    if (LOG_ON) {
        std::cout << "model output num: " << outputNum_ << " output size " << *outputSizeS_ << std::endl;
    }

    inputMluPtrS = (void**)malloc(sizeof(void*) * inputNum_);
    for (int k = 0; k < inputNum_; k++){
        cnrtMalloc(&(inputMluPtrS[k]), inputSizeS_[k]);
    }
    outputMluPtrS = (void**)malloc(sizeof(void*) * outputNum_);
    for (int j = 0; j < outputNum_; j++){
        cnrtMalloc(&(outputMluPtrS[j]), outputSizeS_[j]);
    }

    CNRT_CHECK(cnrtCreateQueue(&queue_));
    CNRT_CHECK(cnrtCreateNotifier(&start_notifier_));
    CNRT_CHECK(cnrtCreateNotifier(&end_notifier_));

    int *dimValues = nullptr;
    int dimNum = 0;
    CNRT_CHECK(cnrtGetInputDataShape(&dimValues, &dimNum, 0, function_));
    in_n_ = dimValues[0];
    in_h_ = dimValues[1];
    in_w_ = dimValues[2];
    in_c_ = dimValues[3];
    free(dimValues);
    if (LOG_ON) {
        std::cout << "model input dimNum: " << dimNum << " N: " << in_n_ << " H: " << in_h_
                  << " W: " << in_w_ << " C: " << in_c_ << std::endl;
    }
    int input_size = in_n_ * in_h_ * in_w_ * in_c_;
    cpu_data_cast_type = malloc(cnrtDataTypeSize(CNRT_FLOAT16) * input_size);
    cpu_data_ = (float *)malloc(input_size * sizeof(float));


    out_n_ = (int *)malloc(sizeof(int) * outputNum_);
    out_h_ = (int *)malloc(sizeof(int) * outputNum_);
    out_w_ = (int *)malloc(sizeof(int) * outputNum_);
    out_c_ = (int *)malloc(sizeof(int) * outputNum_);
    output_data_cast = (void **)malloc(sizeof(void **) * outputNum_);
    outputCpu = (float**)malloc(sizeof(float*) * outputNum_);
    for (int j = 0; j < outputNum_; j++){
        CNRT_CHECK(cnrtGetOutputDataShape(&dimValues, &dimNum, j, function_));
        out_n_[j] = dimValues[0];
        out_h_[j] = dimValues[1];
        out_w_[j] = dimValues[2];
        out_c_[j] = dimValues[3];
        int out_count_ = out_n_[j] * out_h_[j] * out_w_[j] * out_c_[j];
        free(dimValues);
        if (LOG_ON) {
            std::cout << "model output " << j << " dimNum: " << dimNum
                      << " N: " << out_n_[j] << " H: " << out_h_[j]
                      << " W: " << out_w_[j] << " C: " << out_c_[j] << " outputSize "
                      << outputSizeS_[j] << std::endl;
        }

        output_data_cast[j] = malloc(sizeof(float) * out_count_);
        outputCpu[j] = (float*)malloc(cnrtDataTypeSize(CNRT_FLOAT32) * out_count_);
    }
}

void ClassifierLauncher::run_network(cv::Mat &image, std::vector<std::vector<float>> &detections) {
    return;
    if (LOG_ON) {
        std::cout << "image w x h x c: " << image.cols << " x " << image.rows
                  << " x " << image.channels() << std::endl;
    }

    cv::Mat img;
    cv::resize(image, img, cv::Size(in_w_, in_h_));
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3);

    int channels = img.channels();
    for(int k = 0; k < channels; k++){
        float *cpu_data_ptr = cpu_data_ + k * img.rows * img.cols;
        int kk = channels - 1 - k;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                cpu_data_ptr[i * img.cols + j] = (img_float.at<cv::Vec3f>(i, j)[kk] - 127.5F) / 255.0F;
            }
        }
    }
    //for(int i = 0; i < 20; ++i) printf("%d %f\n", i, cpu_data_[i]);
    int dim_values[4] = {in_n_, in_c_, in_h_, in_w_};
    int dim_order[4] = {0, 2, 3, 1};  // NCHW --> NHWC

    CNRT_CHECK(cnrtTransOrderAndCast(cpu_data_, CNRT_FLOAT32,
                                     cpu_data_cast_type, CNRT_FLOAT16,
                                     nullptr, 4, dim_values, dim_order));
    //CNRT_CHECK(cnrtCastDataType(cpu_data_, CNRT_FLOAT32,
    //                            cpu_data_cast_type, CNRT_FLOAT16, input_size, nullptr));
    CNRT_CHECK(cnrtMemcpy(inputMluPtrS[0], cpu_data_cast_type,
                          inputSizeS_[0], CNRT_MEM_TRANS_DIR_HOST2DEV));

    void* param[inputNum_ + outputNum_];
    for (int j = 0; j < inputNum_; j++) {
        param[j] = inputMluPtrS[j];
    }
    for (int j = 0; j < outputNum_; j++) {
        param[inputNum_ + j] = outputMluPtrS[j];
    }

    CNRT_CHECK(cnrtInvokeRuntimeContext(runtime_ctx_, param, queue_, nullptr));

    if (cnrtSyncQueue(queue_) == CNRT_RET_SUCCESS) {
        //std::cout << "SyncStream success" << std::endl;
    } else {
        std::cout << "SyncStream error" << std::endl;
    }

    for (int j = 0; j < outputNum_; j++){
        int out_count_ = out_n_[j] * out_h_[j] * out_w_[j] * out_c_[j];
        CNRT_CHECK(cnrtMemcpy(output_data_cast[j], outputMluPtrS[j],
                              outputSizeS_[j], CNRT_MEM_TRANS_DIR_DEV2HOST));
        CNRT_CHECK(cnrtCastDataType(output_data_cast[j], CNRT_FLOAT16,
                                    outputCpu[j], CNRT_FLOAT32, out_count_, nullptr));
        //continue;
        int mask_size = out_w_[j] * out_h_[j];
        int output_size = out_n_[j] * out_h_[j] * out_w_[j] * out_c_[j];
        uchar *mask_data = (uchar *)malloc(mask_size * sizeof(char));
        for(int i = 0; i < mask_size; i++) {
            //mask_data[i] = outputCpu[j][i*2+1] * 255.0F;
            /*
            if(outputCpu[j][i*2+1] > 0.5F){
                mask_data[i] = 255;
                //printf("%d %f\n", i, outputCpu[j][i*2+1]);
            } else {
                mask_data[i] = 0;
            }
            */
        }
        cv::Mat mask(in_h_, in_w_, CV_8UC1, mask_data);
        cv::imwrite("mask.jpg", mask);
        free(mask_data);
    }
    return;
}

void *ssd_run(void* args) {
    int thread_index = *((int *)args);
    std::string offmodel = "model/unet.cambricon";
    std::string offmodel_tf = "/home/Cambricon-MLU270/tensorflow/src/tensorflow/tensorflow/cambricon_examples/tf_unet/body_condition_score/model/offline.cambricon";
    std::string images_file = "VISIBLE_PIC_41.jpg";
    ClassifierLauncher* launcher = new ClassifierLauncher(offmodel_tf);
    cv::Mat image_input = cv::imread(images_file);

    std::vector<std::vector<float>> detections;
    launcher->run_network(image_input, detections);
    printf("%d thread start\n", thread_index);
    double start = what_time_is_it_now();
    int times = 1;
    for(int i =  0; i < times; i++){
        detections.clear();
        launcher->run_network(image_input, detections);
    }
    double end = what_time_is_it_now();
    printf("%d thread end, spend %f, %f\n", thread_index, end - start, times / (end - start));
    delete launcher;
    return NULL;
}

void ssd_run_voc() {
    std::vector<std::string> image_path;
    char *line = (char *)malloc(sizeof(char) * 1024);
    size_t len = 0;
    ssize_t read;

    FILE * fp = fopen("test_voc2007.txt", "r");
    if (fp == NULL) return;
    while ((read = getline(&line, &len, fp)) != -1) {
        line[read-1] = '\0';
        //printf("Retrieved line of length %zu %s", read, line);
        std::string line_str(line);
        image_path.push_back(line_str);
    }
    fclose(fp);
    free(line);

    std::string offmodel = "models/ssdvgg300.cambricon";
    ClassifierLauncher* launcher = new ClassifierLauncher(offmodel);
    double start = what_time_is_it_now();
    int times = image_path.size();

    std::vector<std::vector<float>> detections;
    std::string outputdir = "output/";
    const char *label_to_name[] = {"background","aeroplane","bicycle","bird","boat","bottle",
                             "bus","car","cat","chair","cow","diningtable","dog","horse",
                             "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};
    for(int k = 0; k < image_path.size(); k ++){
        std::cout << image_path[k] << std::endl;
        cv::Mat image = cv::imread(image_path[k]);
        detections.clear();
        launcher->run_network(image, detections);

        std::string image_name = image_path[k].substr(image_path[k].find_last_of("/") + 1);
        std::string result_name = image_name.replace(image_name.end() - 4, image_name.end(), ".txt");
        std::string output_name = outputdir + result_name;
        std::ofstream output_file(output_name);

        for(int i = 0; i < detections.size(); i++){
            output_file << label_to_name[(int)detections[i][1]] << " " << detections[i][2] << " "
                        << detections[i][3] << " " << detections[i][4] << " "
                        << detections[i][5] << " " << detections[i][6] << std::endl;
        }
        output_file.close();
    }
    double end = what_time_is_it_now();
    printf("spend %f, %f\n", end - start, times / (end - start));
    delete launcher;
}

int main(int argc, char* argv[]) {
    unsigned int real_dev_num;
    cnrtInit(0);
    cnrtGetDeviceCount(&real_dev_num);
    if (real_dev_num == 0) {
        std::cerr << "only have " << real_dev_num << " device(s) " << std::endl;
        cnrtDestroy();
        return -1;
    }

    int get_time = 1;
    if(get_time == 1){
        double start = what_time_is_it_now();
        int thread_num = 1;
        int thread_index[1024];
        std::vector<pthread_t> thread_ids(thread_num);
        for(int i = 0; i < thread_num; i++){
            thread_index[i] = i;
            pthread_create(&thread_ids[i], NULL, ssd_run, &thread_index[i]);
        }

        for(int i = 0; i < thread_num; i++){
            pthread_join(thread_ids[i], NULL);
        }

        double end = what_time_is_it_now();
        printf("main spend %f %f", end - start, (thread_num * 500) / (end - start));
    } else {
        ssd_run_voc();
    }
    cnrtDestroy();
    return 0;
}
