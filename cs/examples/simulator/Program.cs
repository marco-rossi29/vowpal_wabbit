using System;

namespace simulator
{
    class Program
    { 
        static void Main(string[] args)
        {
            int numActions, numContexts, tot_iter, mod_iter, rnd_seed, swap_preferences_iter;
            float minP, maxP, noClickCost, clickCost;

            string ml_args = args[0];
            if (!int.TryParse(args[1], out numActions) ||
                !int.TryParse(args[2], out numContexts) ||
                !float.TryParse(args[3], out minP) ||
                !float.TryParse(args[4], out maxP) ||
                !float.TryParse(args[5], out noClickCost) ||
                !float.TryParse(args[6], out clickCost) ||
                !int.TryParse(args[7], out tot_iter) ||
                !int.TryParse(args[8], out mod_iter) ||
                !int.TryParse(args[9], out rnd_seed) ||
                !int.TryParse(args[10], out swap_preferences_iter))
            {
                Console.WriteLine("Failed to parse input arguments!");
                Console.WriteLine("Usage: simulator.exe ml_args num_actions num_contexts minP maxP noClickCost clickCost tot_iter mod_iter rnd_seed swap_preferences_iter");
                return;
            }

            VowpalWabbitSimulator.Run(ml_args, tot_iter, mod_iter, rnd_seed, numContexts, numActions, minP, maxP, noClickCost, clickCost, swap_preferences_iter);
        }
    }
}
