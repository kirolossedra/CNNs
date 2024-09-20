repeatBranch = [
    imageInputLayer([90 189 1],"Normalization", "none")

    convolution2dLayer(3,8,"Padding",1)
    batchNormalizationLayer
    reluLayer   
    additionLayer(2)
    maxPooling2dLayer(2,"Stride",2)

    
    

    convolution2dLayer(3,16,"Padding",1)
    batchNormalizationLayer
    reluLayer
    additionLayer(2)
    maxPooling2dLayer(2,"Stride",2)


    convolution2dLayer(3,32,"Padding",1)
    batchNormalizationLayer
    reluLayer
    additionLayer(2)
    maxPooling2dLayer(2,"Stride",2)

    


    
    convolution2dLayer(3,64,"Padding",1)
    batchNormalizationLayer
    reluLayer
     
    maxPooling2dLayer(2,"Stride",2)


   
    ];

mainBranch = [
    additionLayer(3)
    fullyConnectedLayer(12)
    softmaxLayer
    ];







misoCNN = dlnetwork();
misoCNN = addLayers(misoCNN, repeatBranch);
misoCNN = addLayers(misoCNN, repeatBranch);
misoCNN = addLayers(misoCNN, repeatBranch);
misoCNN = addLayers(misoCNN, mainBranch);


misoCNN = connectLayers(misoCNN, "imageinput", "addition_1/in2");

misoCNN = connectLayers(misoCNN, "conv_2", "addition_2/in2");


misoCNN = connectLayers(misoCNN, "conv_3", "addition_3/in2");






misoCNN = connectLayers(misoCNN, "imageinput_1", "addition_4/in2");

misoCNN = connectLayers(misoCNN, "conv_6", "addition_5/in2");


misoCNN = connectLayers(misoCNN, "conv_7", "addition_6/in2");




misoCNN = connectLayers(misoCNN, "imageinput_2", "addition_7/in2");

misoCNN = connectLayers(misoCNN, "conv_10", "addition_8/in2");


misoCNN = connectLayers(misoCNN, "conv_11", "addition_9/in2");









misoCNN = connectLayers(misoCNN, "maxpool_4", "addition/in1");


misoCNN = connectLayers(misoCNN, "maxpool_8", "addition/in2");
misoCNN = connectLayers(misoCNN, "maxpool_12", "addition/in3");
