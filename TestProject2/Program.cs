// See https://aka.ms/new-console-template for more information
// for the projection
using System.Security.Cryptography.X509Certificates;

namespace SignalStrengthMap
{
    public class Datapreparation
    {

        
        public static double[] MercatorProjection(double lon, double lat)    // method
        {
            int rMajor=6378137;
            double x = rMajor * lon * (Math.PI / 180.0);
            double y = rMajor * Math.Log(
                Math.Tan(
                    Math.PI/4 + lon * (Math.PI / 180.0)/2
                )
            );
            double[] xy = {x,y};

            return xy;
        }

        public static double[,] MercatorProjectionArray(double[] lon, double[] lat)
        {
            // int rMajor = 6378137;
            int length = lon.Length;
            double[,] xyArray= new double[length,2];
            // double[] xs = new double[length];
            // double[] ys = new double[length];
            for (int i = 0; i < length; i++)
            {
                double[] temp=MercatorProjection(lon[i],lat[i]);
                xyArray[i,0]=temp[0];
                xyArray[i,1]=temp[1];
            }
            
            
            return xyArray;
        }
    }
     
     internal class Program
     {
        static void Main(string[] args)
        {
            Datapreparation cal = new Datapreparation();
            double lonTest=-79.390662;
            double latTest=43.733180;
            double[] xyCordinate = Datapreparation.MercatorProjection(lonTest,latTest);

            double[] lonTestArray={-79.390662, -80};
            double[] latTestArray={43.733180,45};
            double[,] xyCordinateArray = Datapreparation.MercatorProjectionArray(lonTestArray,latTestArray); 

            String traindata = "Data//synthetic_train_800.txt";
            double[][] trainX = Utils.MatLoad(traindata,
              new int[] { 0, 1 }, ',', "#");
            double[] trainY = Utils.MatToVec(Utils.MatLoad(traindata,
              new int[] { 2 }, ',', "#"));
            int trainLen= trainX.Length;

            // convert the latitude and longitude to X and Y
            double[][] xyCordinateTrain = Utils.MatCreate(trainLen,2);
            int gridLen = 100000;

            for (int i = 0; i < trainLen; i++)
            {
                double[] temp=Datapreparation.MercatorProjection(trainX[i][0],trainX[i][1]);
                xyCordinateTrain[i] = temp;
            }

            int kValue=5;
            KNNR model = new KNNR(kValue, "skewed");
            model.Store(xyCordinateTrain, trainY);  // no need after first

            // split the how area into grids
            double minxCordinate = Double.PositiveInfinity;
            double minyCordinate = Double.PositiveInfinity;
            double maxxCordinate = Double.NegativeInfinity;
            double maxyCordinate = Double.NegativeInfinity;

            for (int i = 0; i < trainLen; i++)
            {
                if (xyCordinateTrain[i][0]>maxxCordinate)
                {
                    maxxCordinate=xyCordinateTrain[i][0];
                }

                if (xyCordinateTrain[i][0]<minxCordinate)
                {
                    minxCordinate=xyCordinateTrain[i][0];
                }

                if (xyCordinateTrain[i][1]>maxyCordinate)
                {
                    maxyCordinate=xyCordinateTrain[i][1];
                }

                if (xyCordinateTrain[i][1]<minyCordinate)
                {
                    minyCordinate=xyCordinateTrain[i][1];
                }
            }

            double xAxisLen = maxxCordinate - minxCordinate;
            double yAxisLen = maxyCordinate - minyCordinate;
            int xAxisNum = (int)(xAxisLen/gridLen);
            int yAxisNum = (int)(yAxisLen/gridLen);
            double xgridLen = xAxisLen/xAxisNum;
            double ygridLen = yAxisLen/yAxisNum;

            Console.WriteLine("xAxisLen: "+xAxisLen);
            Console.WriteLine("yAxisLen: "+yAxisLen);
            Console.WriteLine("xAxisNum: "+xAxisNum);
            Console.WriteLine("yAxisNum: "+yAxisNum);
            
            // create an matrix to store the prediction result
            double[][] evenPointsPredition = Utils.MatCreate(xAxisNum,yAxisNum);
            for (int i =0; i < xAxisNum;i++)
            {
                for (int j =0;j < yAxisNum;j++)
                {
                    double[] gridCenter = {
                        i*xgridLen+minxCordinate,
                        j*ygridLen+minyCordinate
                    };
                    evenPointsPredition[i][j] = model.Predict(gridCenter);
                }
            }
        }
    }

    
}