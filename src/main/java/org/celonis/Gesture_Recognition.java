package org.celonis;

import java.io.File;
import java.util.List;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @ Md. Rezaul Karim
 * 17.04.2022
 *
 */
public class Gesture_Recognition {
    public static void main(String[] args) throws Exception {
        File file = new File("C:/Users/admin-karim/Downloads/Gesture_detection/src/main/resources/gesture.csv");
        RecordReader reader = new CSVRecordReader(',');
        reader.initialize(new FileSplit(file));
        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, 128, 945, 8);
        
        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();        
        preProcessor.fit(iterator);
        iterator.setPreProcessor(preProcessor);
        
        DataSet iriDataSet = iterator.next();
        iriDataSet.shuffle();
        
        // 80 % for training
        SplitTestAndTrain testAndTrain = iriDataSet.splitTestAndTrain(0.8);
        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();
        
        OutputLayer outputLayer = new OutputLayer.Builder()
            .nIn(945).nOut(8)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)              
            .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
             .seed(123)
             .weightInit(WeightInit.XAVIER)
             .updater(new Nesterovs(0.1, 0.9))
             .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
             .list()
             .layer(outputLayer)
             .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Logger log = LoggerFactory.getLogger(Gesture_Recognition.class);
        for (int i =0; i< 1000; i++){
            model.fit(trainSet);
            log.info("Score on epoch {} : {} ", i, model.score()); 
        }

        List<DataSet> list = testSet.asList();    
        DataSetIterator testIterator = new ListDataSetIterator<>(list); 
        Evaluation eval = model.evaluate(testIterator);
        
        log.info("Precision : {}",eval.precision()); 
        log.info("Recall : {}",eval.recall()); 
        log.info("Accuracy : {}", eval.accuracy()) ;  
        log.info("\n-----Confusion matrix-----\n") ;
        log.info("{}",eval.confusionMatrix());         
    }
}