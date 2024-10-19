using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        // Creating MLContext
        var mlContext = new MLContext();

        // Path to data
        string dataPath = "spamData.csv";
        
        // Load data
        IDataView dataView = mlContext.Data.LoadFromTextFile<EmailData>(dataPath, separatorChar: ',', hasHeader: true);

        // Pipeline
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.EmailText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

        // Train the model
        var model = pipeline.Fit(dataView);  
    }
}

// data and predictions class
public class EmailData
{
    [LoadColumn(0)] // First column in the CSV
    public bool Label { get; set; }

    [LoadColumn(1)] // Second column in the CSV
    public string? EmailText { get; set; }
}

public class EmailPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsSpam { get; set; }
}
