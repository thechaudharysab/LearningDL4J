package com.ibjects.ComputerVision.SignLanguage;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Random;

public class SignLanguageClassification {

    private static int epochs = 10; //120
    private static int batchSize = 32;
    private static int seed = 123;
    private static int numClasses = 10;

    private static int height = 400;
    private static int width = 400;
    private static int channel = 3;

    private static final Random randNumGen = new Random(seed);

    public static void main(String[] args) throws IOException {

        BasicConfigurator.configure();
        buildModel();

    }

    private static void buildModel() throws IOException {

        SignLanguageDataIteration.setup(batchSize, 80);

        DataSetIterator trainIter = SignLanguageDataIteration.trainIterator();
        DataSetIterator testIter = SignLanguageDataIteration.testIterator();

        //model configuration
        double nonZeroBias = 0.1;
        double dropOut = 0.5;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed).weightInit(WeightInit.XAVIER).activation(Activation.RELU).updater(new Adam(0.001))
                .convolutionMode(ConvolutionMode.Same).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(5*1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(
                        new int[]{11,11}, new int[]{4,4})
                .name("cnn1").convolutionMode(ConvolutionMode.Truncate).nIn(channel).nOut(96).build())
                .layer(1, new LocalResponseNormalization.Builder().build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .name("cnn2")
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .name("maxpool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .name("cnn3")
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn4")
                        .nOut(384)
                        .dropOut(0.2)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn5")
                        .nOut(256)
                        .dropOut(0.2)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
                        .name("maxpool3")
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output").nOut(numClasses).activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER).build())
                .setInputType(InputType.convolutional(height, width, channel))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        System.out.println(model.summary());

    }
}
