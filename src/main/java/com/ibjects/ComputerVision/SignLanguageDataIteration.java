package com.ibjects.ComputerVision;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;

public class SignLanguageDataIteration {

    private static final int height = 400;
    private static final int width = 400;
    private static final int channels = 3;
    private static final int numClasses = 36;

    //Images are of format given by allowedExtension
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final String DATA_URL = "";

    //Random number generator
    private static final Random rng  = new Random(123);

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public SignLanguageDataIteration() {
        //Empty Constructor
    }

    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        if (training && transform != null) {
            recordReader.initialize(split, transform);
        } else {
            recordReader.initialize(split);
        }

        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor(scaler);

        return iter;

    }

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData, false);
    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {

        batchSize = batchSizeArg;

        File parentDir = new File("src/main/resources");
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];

    }

    private static void loadAndUnzipFile() throws IOException {

        String outputPath = "src/main/resources"; //Right click on resources -> copy -> Absolute path
        File zipFile = new ClassPathResource("asl_dataset.zip").getFile();
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), outputPath);
    }



}
