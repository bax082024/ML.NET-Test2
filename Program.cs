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






  }
}




