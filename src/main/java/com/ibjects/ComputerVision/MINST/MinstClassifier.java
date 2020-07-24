package com.ibjects.ComputerVision.MINST;

import org.apache.log4j.BasicConfigurator;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MinstClassifier {

    private static final String RESOURCES_FOLDER_PATH = "/Users/chaudhrytalha/Documents/LearningDL4J/src/main/resources/mnist_png/";

    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;

    private static final int N_SAMPLES_TRAINING = 60000;
    private static final int N_SAMPLES_TESTING = 10000;

    private static final int N_OUTCOMES = 10;

    public static void main(String[] args) throws IOException {

        BasicConfigurator.configure();

        long t0 = System.currentTimeMillis();
        DataSetIterator dsi = getDataSetIterator(RESOURCES_FOLDER_PATH + "training", N_SAMPLES_TRAINING);

        int rndSeed = 123;
        int nEpochs = 2;

        System.out.printf("Build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rndSeed).updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4).list()
                .layer(new DenseLayer.Builder().nIn(HEIGHT*WIDTH).nOut(1000).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(1000)
                        .nOut(N_OUTCOMES).activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(dsi);
            System.out.println("Completed Epoch: " + i);
            dsi.reset();
            DataSetIterator testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH + "testing", N_SAMPLES_TESTING);
            System.out.println("Evaluate model...");
            Evaluation evaluation = model.evaluate(testDsi);
            System.out.println(evaluation.stats());
            long t1 = System.currentTimeMillis();
            double t = (double)(t1-t0)/1000.0;
            System.out.println("\n\nTotal time: "+t+" seconds");
        }
    }

    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        File folder = new File(folderPath);
        File[] digitFolders = folder.listFiles();
        NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scalar = new ImagePreProcessingScaler(0,1);

        INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT*WIDTH});
        INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

        int n = 0;
        for (File digitFolder: digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();

            for (File imageFile : imageFiles) {
                INDArray img = nil.asRowVector(imageFile);
                scalar.transform(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }
        }

        DataSet dataSet = new DataSet(input, output);
        List<DataSet> listDataSet = dataSet.asList();
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
        int batchSize = 10;

        DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);
        return dsi;
    }



}
