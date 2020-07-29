using System;
using System.Collections.Generic;

namespace stats
{
    class Program
    {
        static void Main(string[] args)
        {

            List<double> numbers = new List<double> { -5, 10, 15 };

            Console.WriteLine(StatsCalculator.CalculateStandardDeviation(numbers));

        }
    }
}
