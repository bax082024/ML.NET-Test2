using Microsoft.ML.Data;
public class EmailData
{
  public bool Label { get; set; }
  public string EmailText { get; set; }
}

public class EmailPrediction
{
  [ColumnName("PredictedLabel")]
  public bool IsSpam { get; set; }
}