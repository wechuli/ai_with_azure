using System;
using System.Collections.Generic;

namespace stats
{

    public static class StatsCalculator
    {

        public static double CalculateStandardDeviation(ICollection<double> dataset)
        {

            double sumOfSquares = 0;

            double mean = CalculateMean(dataset);

            Console.WriteLine(mean);
            foreach (var item in dataset)
            {
                sumOfSquares += (Math.Pow((mean - item), 2));
            }

            return Math.Sqrt(sumOfSquares / dataset.Count);


        }

        public static double CalculateMean(ICollection<double> dataset)
        {
            double sum = 0;
            foreach (var item in dataset)
            {
                sum += item;

            }

            return sum / dataset.Count;
        }


    }

}