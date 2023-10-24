using System;
using System.IO;

namespace KNNregression
{
  internal class KNNregressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin weighted k-NN regression ");

      // 1. load data into memory
      Console.WriteLine("\nLoading train and test data ");
      string trainFile = "Data//synthetic_train.txt";
      double[][] trainX = Utils.MatLoad(trainFile,
        new int[] { 0, 1, 2, 3, 4 }, ',', "#");
      double[] trainY = Utils.MatToVec(Utils.MatLoad(trainFile,
        new int[] { 5 }, ',', "#"));

      string testFile = "Data//synthetic_test.txt";
      double[][] testX = Utils.MatLoad(testFile,
        new int[] { 0, 1, 2, 3, 4 }, ',', "#");
      double[] testY = Utils.MatToVec(Utils.MatLoad(testFile,
        new int[] { 5 }, ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        Utils.VecShow(trainX[i], 4, 9, true);

      Console.WriteLine("\nFirst three target y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4"));

      // 2. find a good k value
      double accTrain = 0.0;
      double accTest = 0.0;
      double rmseTrain = 0.0;
      double rmseTest = 0.0;

      Console.WriteLine("\nExploring k values (acc " +
        "within 0.15) ");
      Console.WriteLine("");
      int[] candidates = new int[] { 1, 3, 5, 7 };
      foreach (int k in candidates)
      {
        Console.Write("k = " + k);
        KNNR model = new KNNR(k, "skewed");
        model.Store(trainX, trainY);  // no need after first
        accTrain = Accuracy(model, trainX, trainY, 0.15);
        accTest = Accuracy(model, testX, testY, 0.15);
        rmseTrain = RootMSE(model, trainX, trainY);
        rmseTest = RootMSE(model, testX, testY);

        Console.Write("  Acc train = " +
          accTrain.ToString("F4"));
        Console.Write("  Acc test = " +
          accTest.ToString("F4"));
        Console.Write("  RMSE train = " +
          rmseTrain.ToString("F4"));
        Console.Write("  RMSE test = " +
          rmseTest.ToString("F4"));
        Console.WriteLine("");
      }

      // 3. create and pseudo-train model
      Console.WriteLine("\nCreating k-NN regression" +
        " model k = 5 weighting = skewed ");
      KNNR knnr_model = new KNNR(5, "skewed");
      Console.WriteLine("Done ");

      Console.WriteLine("\nStoring train data into model ");
      knnr_model.Store(trainX, trainY);
      Console.WriteLine("Done ");

      // 4. evaluate model
      Console.WriteLine("\nComputing accuracy (within 0.15) ");
      accTrain = Accuracy(knnr_model, trainX, trainY, 0.15);
      Console.WriteLine("Accuracy train = " +
        accTrain.ToString("F4"));
      accTest = Accuracy(knnr_model, testX, testY, 0.15);
      Console.WriteLine("Accuracy test = " +
        accTest.ToString("F4"));

      // 5. use model
      Console.WriteLine("\nExplaining for " +
        "x = (0.5, -0.5, 0.5, -0.5, 0.5) ");
      double[] x = 
        new double[] { 0.5, -0.5, 0.5, -0.5, 0.5 };
      knnr_model.Explain(x);
      
      // 6. TODO: save model to file

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main

    static double Accuracy(KNNR model, double[][] dataX,
       double[] dataY, double pctClose)
    {
      int numCorrect = 0; int numWrong = 0;
      int n = dataX.Length;
      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double actualY = dataY[i];
        double predY = model.Predict(x);
        
        if (Math.Abs(actualY - predY) < 
          Math.Abs(pctClose * actualY))
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / n;
    }

    static double RootMSE(KNNR model, double[][] dataX,
       double[] dataY)
    {
      double sum = 0.0;
      int n = dataX.Length;
      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double actualY = dataY[i];
        double predY = model.Predict(x);
        sum += (actualY - predY) * (actualY - predY);
      }

      return Math.Sqrt(sum / n);
    } // RootMSE

  } // Program

  public class KNNR
  {
    public int k;
    public double[][]? trainX;
    public double[]? trainY;
    public string weighting;

    public KNNR(int k, string weighting)
    {
      this.k = k;
      this.trainX = null;
      this.trainY = null;
      this.weighting = weighting;
      // 'uniform', 'skewed'
    }

    public void Store(double[][] trainX, double[] trainY)
    {
      this.trainX = trainX;  // by ref
      this.trainY = trainY;
    }

    public double Predict(double[] x)
    {
      if (this.trainX == null)
        Console.WriteLine("Error: Store() not yet called ");

      // 0. set up ordering/indices
      int n = this.trainX.Length;
      int[] indices = new int[n];
      for (int i = 0; i < n; ++i)
        indices[i] = i;

      // 1. compute distances from x to all trainX
      double[] distances = new double[n];
      for (int i = 0; i < n; ++i)
        distances[i] = EucDistance(x, this.trainX[i]);

      // 2. sort distances, indices of X and Y, by distances
      Array.Sort(distances, indices);

      // 3. return weighted first k sorted trainY values
      double[]? wts = null;
      if (this.weighting == "uniform") // .2 .2 .2 .2 .2
        wts = UniformWts(this.k);
      else if (this.weighting == "skewed") // .3 .2 .2 .2 .1
        wts = SkewedWts(this.k);

      double result = 0.0;
      for (int i = 0; i < this.k; ++i)
        result += wts[i] * this.trainY[indices[i]];

        return result;
    } // Predict

    public void Explain(double[] x)
    {
      // 0. set up ordering/indices
      int n = this.trainX.Length;
      int[] indices = new int[n];
      for (int i = 0; i < n; ++i)
        indices[i] = i;

      // 1. compute distances from x to all trainX
      double[] distances = new double[n];
      for (int i = 0; i < n; ++i)
        distances[i] = EucDistance(x, this.trainX[i]);

      // 2. sort distances, indices of X and Y, by distances
      Array.Sort(distances, indices);

      // 3. compute weighted first k sorted trainY values
      double[]? wts = null;
      if (this.weighting == "uniform")
        wts = UniformWts(this.k);
      else if (this.weighting == "skewed")
        wts = SkewedWts(this.k);

      double result = 0.0;
      for (int i = 0; i < this.k; ++i)
        result += wts[i] * this.trainY[indices[i]];

      // 4. show info 
      for (int i = 0; i < this.k; ++i)
      {
        int j = indices[i];
        Console.Write("X = ");
        Console.Write("[" + j.ToString().
          PadLeft(3) + "] ");
        Utils.VecShow(this.trainX[j], 4, 9, false);
        Console.Write(" | y = ");
        Console.Write(this.trainY[j].ToString("F4"));
        Console.Write(" | dist = ");
        Console.Write(distances[i].ToString("F4"));
        Console.Write(" | wt = ");
        Console.Write(wts[i].ToString("F4"));
        Console.WriteLine("");
      }
      
      Console.WriteLine("\nPredicted y = " +
        result.ToString("F4"));

      // show fancy calculation for predicted y
      //Console.WriteLine("\nPredicted y = ");
      //for (int i = 0; i < this.k; ++i)
      //{
      //  Console.Write("(" +
      //    trainY[indices[i]].ToString("F4") +
      //    " * " + wts[i].ToString("F3") + ")");
      //  if (i < this.k-1)
      //    Console.Write(" + ");
      //}
      //Console.WriteLine("\n= " + result.ToString("F4"));


    } // Explain

    private static double EucDistance(double[] v1,
      double[] v2)
    {
      int dim = v1.Length;
      double sum = 0.0;
      for (int j = 0; j < dim; ++j)
        sum += (v1[j] - v2[j]) * (v1[j] - v2[j]);
      return Math.Sqrt(sum);
    }

    private static double[] UniformWts(int k)
    {
      double[] result = new double[k];
      for (int i = 0; i < k; ++i)
        result[i] = 1.0 / k;
      return result;
    }

    private static double[] SkewedWts(int k)
    {
      double[] result = new double[k];
      if (k == 1) result[0] = 1.0;
      else if (k == 2)
      {
        result[0] = 0.6000;
        result[1] = 0.4000;
      }
      else if (k == 3)
      {
        result[0] = 0.4000;
        result[1] = 0.3500;
        result[2] = 0.2500;
      }
      else if (k >= 4)
      {
        double big = 1.5 * (1.0 / k);  // 1.5 * 0.25 = 0.3750
        double small = 0.5 * (1.0 / k);  // 0.5 * 0.25 = 0.1250
        double remainder = 1.0 - (big + small);  // 0.5000
        double x = remainder / (k - 2);  // 0.2500
        result[0] = big;
        result[k - 1] = small;
        for (int i = 1; i < k - 1; ++i)
          result[i] = x;
      }
      return result;  // 0.3750, 0.2500, 0.2500, 0.1250
    }
    
  } // class KNNR

  public class Utils
  {
    public static double[][] MatCreate(int rows, int cols)
    {
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
      return result;
    }
    static int NumNonCommentLines(string fn,
        string comment)
    {
      int ct = 0;
      string line = "";
      FileStream ifs = new FileStream(fn,
        FileMode.Open);
      StreamReader sr = new StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
        if (line.StartsWith(comment) == false)
          ++ct;
      sr.Close(); ifs.Close();
      return ct;
    }

    public static double[][] MatLoad(string fn,
        int[] usecols, char sep, string comment)
    {
      // count number of non-comment lines
      int nRows = NumNonCommentLines(fn, comment);
      int nCols = usecols.Length;
      double[][] result = MatCreate(nRows, nCols);
      string line = "";
      string[] tokens = null;
      FileStream ifs = new FileStream(fn, FileMode.Open);
      StreamReader sr = new StreamReader(ifs);

      int i = 0;
      while ((line = sr.ReadLine()) != null)
      {
        if (line.StartsWith(comment) == true)
          continue;
        tokens = line.Split(sep);
        for (int j = 0; j < nCols; ++j)
        {
          int k = usecols[j];  // into tokens
          result[i][j] = double.Parse(tokens[k]);
        }
        ++i;
      }
      sr.Close(); ifs.Close();
      return result;
    }

    public static void MatShow(double[][] m,
      int dec, int wid)
    {
      for (int i = 0; i < m.Length; ++i)
      {
        for (int j = 0; j < m[0].Length; ++j)
        {
          double v = m[i][j];
          if (Math.Abs(v) < 1.0e-8) v = 0.0; // hack
          Console.Write(v.ToString("F" +
            dec).PadLeft(wid));
        }
        Console.WriteLine("");
      }
    }

    public static double[] MatToVec(double[][] m)
    {
      int rows = m.Length;
      int cols = m[0].Length;
      double[] result = new double[rows * cols];
      int k = 0;
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
        result[k++] = m[i][j];
      }
      return result;
    }

    public static void VecShow(double[] vec,
      int dec, int wid, bool newLine)
    {
      for (int i = 0; i < vec.Length; ++i)
      {
        double x = vec[i];
        if (Math.Abs(x) < 1.0e-8) x = 0.0;  // hack
        Console.Write(x.ToString("F" +
          dec).PadLeft(wid));
      }
      if (newLine == true)
        Console.WriteLine("");
    }

    public static void VecShow(int[] vec, int wid,
      bool newLine)
    {
      for (int i = 0; i < vec.Length; ++i)
      {
        int x = vec[i];
        Console.Write(x.ToString().PadLeft(wid));
      }
      if (newLine == true)
        Console.WriteLine("");
    }

  } // class Utils

} // ns