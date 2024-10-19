using System;
using System.Data;
using System.Security.Cryptography.X509Certificates;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
  static void Main(string[] args)
  {
    // Path to data
    var mlContext = new MLContext();
    {
      // Path to data
      string dataPath = "spamData.csv";
      IDataView dataView = mlContext.Data.LoadFromTextFile<EmailData>(dataPath, separatorChar: ',', hasHeader: true);

    }

    // Pipeline
    var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.EmailText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName:"Features"));

    var model = pipeline.Fit(dataView);

    var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, EmailPrediction>(model);

    






  }
}




