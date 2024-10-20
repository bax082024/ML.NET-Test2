using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        // Create MLContext
        var mlContext = new MLContext();

        // Path to data
        string dataPath = "spamData.csv";

        // Load data from CSV file
        IDataView dataView = mlContext.Data.LoadFromTextFile<EmailData>(dataPath, separatorChar: ',', hasHeader: true);

        // Split data into training (50%) and testing (50%). hopefully better result 
        var splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.5);

        // Define the pipeline
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmailData.EmailText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

        // Train the model on the training data
        var model = pipeline.Fit(splitData.TrainSet);

        // Evaluate the model using the testing set
        var testSetPredictions = model.Transform(splitData.TestSet);
        var metrics = mlContext.BinaryClassification.Evaluate(testSetPredictions, labelColumnName: "Label");

        // Output evaluation metrics
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

        // Create a prediction engine for testing new emails
        var predictionEngine = mlContext.Model.CreatePredictionEngine<EmailData, EmailPrediction>(model);

        // Test with a sample email
        var testEmail = new EmailData { EmailText = "You have won a free iPhone!" };
        var prediction = predictionEngine.Predict(testEmail);

        // Print the prediction result
        Console.WriteLine($"Prediction for the test email: {(prediction.IsSpam ? "Spam" : "Not Spam")}");
    }
}

// Classes for email data and predictions
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
