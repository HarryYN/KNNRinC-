using System;
using System.IO;


namespace SignalStrengthMap
{
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
}