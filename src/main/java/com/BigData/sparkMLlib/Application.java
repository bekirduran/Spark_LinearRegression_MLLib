package com.BigData.sparkMLlib;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.mllib.feature.VectorTransformer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class Application {
    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "C:\\hadoop-common-2.2.0-bin-master");
        SparkSession sparkSession = SparkSession.builder().appName("Spark Machine Learning Examples")
                .master("local").getOrCreate();

        StructType myStructType = new StructType()
                .add("No","integer")
                .add("X1 transaction date","double")
                .add("X2 house age","double")
                .add("X3 distance to the nearest MRT station","double")
                .add("X4 number of convenience stores","integer")
                .add("X5 latitude","double")
                .add("X6 longitude","double")
                .add("Y house price of unit area","double");

        Dataset<Row> rowDataset = sparkSession.read().format("csv")
                .option("header", true)
                .option("interSchema", true)
                .schema(myStructType)
                .load("src/main/Data/Real_estate.csv");

        rowDataset.show();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"No","X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude"})
                .setOutputCol("features");
        Dataset<Row> transform = vectorAssembler.transform(rowDataset);

        Dataset<Row> finaldata = transform.select("features", "Y house price of unit area");

        Dataset<Row>[] randomSplit = finaldata.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingDataSet = randomSplit[0];
        Dataset<Row> testDataSet = randomSplit[1];

        LinearRegression linearRegression = new LinearRegression();
        linearRegression.setLabelCol("Y house price of unit area");

        LinearRegressionModel model = linearRegression.fit(trainingDataSet);

        LinearRegressionTrainingSummary summary = model.summary();
        System.out.println("Model accuracy: "+summary.r2());

        Dataset<Row> transformTest = model.transform(testDataSet);

        transformTest.show();

    }
}
