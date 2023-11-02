  using System;
using System.IO;

namespace SignalStrengthMap
{
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

}