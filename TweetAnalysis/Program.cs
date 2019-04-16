using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;


namespace TweetAnalysis
{
    class Program
    {
        static readonly string _dataPathYelp = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _dataPathIMDB = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labelled.txt");
        static readonly string _dataPathAmazon = Path.Combine(Environment.CurrentDirectory, "Data", "amazon_cells_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static readonly string _tweetsPath = Path.Combine(Environment.CurrentDirectory, "Data", "tweets.txt");
        static readonly string _resultsPath = Path.Combine(Environment.CurrentDirectory, "Data", "results.txt" + DateTime.Now.ToString("yyyyMMddHHmmssfff"));

        static readonly string _inputImageClassifierZip = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
       

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            TrainCatalogBase.TrainTestData splitDataViewYelp = LoadData(mlContext, _dataPathYelp);
            TrainCatalogBase.TrainTestData splitDataViewIMDB = LoadData(mlContext, _dataPathIMDB);
            TrainCatalogBase.TrainTestData splitDataViewAmazon = LoadData(mlContext, _dataPathAmazon);
            ITransformer model = LoadOldModel(mlContext);
            //ITransformer trainedModel = Train(model, mlContext, splitDataViewYelp.TrainSet);
            //trainedModel = Train(trainedModel, mlContext, splitDataViewIMDB.TrainSet);
            //trainedModel = Train(trainedModel, mlContext, splitDataViewAmazon.TrainSet);
            Console.WriteLine("Evaluate on Yelp dataset");
            Evaluate(model, mlContext, splitDataViewYelp.TestSet);

            // <SnippetCallUseLoadedModelWithBatchItems>
            UseLoadedModelWithBatchItems(mlContext);
            // </SnippetCallUseLoadedModelWithBatchItems>

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }


        public static ITransformer LoadOldModel(MLContext mlContext)
        {
            ITransformer loadedModel;
            using (var fileStream = new FileStream(_modelPath, FileMode.Open))
                loadedModel = mlContext.Model.Load(fileStream);

            return loadedModel;

        }

        public static ITransformer Train(ITransformer model,MLContext mlContext, IDataView splitTrainSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 100, numTrees: 30, minDatapointsInLeaves: 10));
               // .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 100, numTrees: 30, minDatapointsInLeaves: 10));
            model = pipeline.Fit(splitTrainSet);
            SaveModelAsFile(mlContext, model);
            return model;
        }

        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext, String _dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }


        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 100, numTrees: 30, minDatapointsInLeaves: 10));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;

        }

        public static void Evaluate(ITransformer model,MLContext mlContext,IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            SaveModelAsFile(mlContext, model);
        }
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            //UseModelWithSingleItem(mlContext, model);
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            var resultprediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {
            string[] myWholeFile = File.ReadAllLines(_tweetsPath);
            List<SentimentData> myList = new List<SentimentData>();
            int currentIndex = 0;
            while (currentIndex < myWholeFile.Length)
            {
                myList.Add(new SentimentData
                {
                    SentimentText = myWholeFile[currentIndex].ToString()
                });
                currentIndex++;
            }
            IEnumerable<SentimentData> sentiments = myList;

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }
            IDataView sentimentStreamingDataView = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach ((SentimentData sentiment, SentimentPrediction prediction) in sentimentsAndPredictions)
            {
                string results = $" Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {String.Format("{0:N7}",prediction.Probability)} | Sentiment: {sentiment.SentimentText} \n";
                Console.WriteLine(results);
                File.AppendAllText(_resultsPath, results);
            }
            Console.WriteLine("=============== End of predictions ===============");
            SaveModelAsFile(mlContext, loadedModel);
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
