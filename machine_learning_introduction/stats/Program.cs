using System;
using System.Collections.Generic;

namespace stats
{
    class Program
    {
        static void Main(string[] args)
        {
            var newds = Guid.NewGuid();

            List<double> numbers = new List<double> { -5, 10, 15 };

            // Console.WriteLine(StatsCalculator.CalculateStandardDeviation(numbers));
            Console.WriteLine(newds);

        }
    }
}
