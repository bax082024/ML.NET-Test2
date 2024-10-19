using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
  static void Main(string[] args)
  {
    var mlContext = new MLContext();
    {
      string dataPath = "spamData.csv";
      IDataView dataView = mlContext.Data.LoadFromTextFile<EmailData>(dataPath, separatorChar: ',', hasHeader: true);
    }

    var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.EmailText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName:"Features"));






  }
}




