# int8 模型中每个 8 位整数 i 表示的实际值为: value = i * 2 ^ position/scale
# scripts/build_caffe.sh
export LD_LIBRARY_PATH=../../build/lib:/usr/local/neuware/lib64

g++ -std=c++11 -I/usr/local/neuware/include offline.cc `pkg-config --cflags --libs opencv` -L/usr/local/neuware/lib64 -lcnrt -lpthread && ./a.out
exit

./../../build/tools/generate_quantized_pt-d -ini_file convert_int8.ini
#exit
sleep 2
#python2 ../../scripts/meanAP_VOC.py test_voc2007.txt output /home/Cambricon-MLU270/datasets/VOC2007/Annotations
../../build/tools/caffe-d genoff  -model model/deploy_mlu_cn.prototxt \
                          -weights model/solver_iter_30000.caffemodel \
                          -mname unet \
                          -mcore MLU270 \
                          -opt_level 1 \
                          -batchsize 1 \
                          -core_number 4 \
                          -simple_compile 1
sleep 2
mv unet.cambricon* model/
exit
../../build/examples/ssd/ssd_offline_singlecore-d \
                  -offlinemodel models/ssdvgg300.cambricon \
                  -images image_list \
                  -outputdir output \
                  -confidencethreshold 0.01 \
                  -labelmapfile ./labelmap_voc.prototxt
exit



g++ -std=c++11 -I/usr/local/neuware/include offline.cc `pkg-config --cflags --libs opencv` -L/usr/local/neuware/lib64 -lcnrt -lpthread

exit


../../build/examples/ssd/ssd_online_singlecore-d \
    -model models/deploy_mlu_cn.prototxt \
    -weights models/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel \
    -images /home/Cambricon-MLU270/caffe/src/.caffe/examples/ssd/file_list_for_release \
    -outputdir output \
    -labelmapfile /home/Cambricon-MLU270/caffe/src/.caffe/examples/ssd/labelmap_voc.prototxt \
    -confidencethreshold 0.01 \
    -mmode MFUS \
    -mcore MLU270 \
    -opt_level 1

exit


../../build/tools/caffe-d genoff -model models/deploy_mlu_cn.prototxt \
    -weights models/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel \
    -mname ssdvgg300 \
    -mcore MLU270 \
    -core_number 1 \
    -batchsize 1 \
    -opt_level 1

exit


g++ -std=c++11 -DUSE_MLU -I../../examples/common/include -I/usr/local/neuware/include/ -I../../build/install/include offline_full_run.cpp `pkg-config --cflags --libs opencv` -L../../build/install/lib -lcaffe-d -lglog -lgflags -L/usr/local/neuware/lib64 -lcnrt -lboost_system
./a.out -offlinemodel lenet.cambricon -images image_list
#../../build/examples/offline_full_run/offline_full_run-d  -offlinemodel lenet.cambricon -images image_list
exit

#exit

../../build/examples/clas_offline_singlecore/clas_offline_singlecore-d \
    -offlinemodel lenet.cambricon \
    -labels synset_words.txt \
    -images image_list

exit


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../build/lib
../../build/examples/clas_online_singlecore/clas_online_singlecore-d \
    -model lenet_cn.prototxt \
    -weights lenet_iter_10000.caffemodel \
    -labels synset_words.txt \
    -images image_list \
    -mcore MLU270 \
    -mmode MLU #CPU
