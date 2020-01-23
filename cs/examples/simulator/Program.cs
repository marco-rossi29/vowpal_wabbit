using System;

namespace simulator
{
    class Program
    { 
        static void Main(string[] args)
        {
            string ml_args = args[0];
            if (!int.TryParse(args[1], out int numActions) ||
                !int.TryParse(args[2], out int numContexts) ||
                !float.TryParse(args[3], out float minP) ||
                !float.TryParse(args[4], out float maxP) ||
                !float.TryParse(args[5], out float noClickCost) ||
                !float.TryParse(args[6], out float clickCost) ||
                !int.TryParse(args[7], out int tot_iter) ||
                !int.TryParse(args[8], out int mod_iter) ||
                !int.TryParse(args[9], out int rnd_seed) ||
                !int.TryParse(args[10], out int swap_preferences_iter))
            {
                Console.WriteLine("Failed to parse input arguments!");
                Console.WriteLine("Usage: simulator.exe ml_args num_actions num_contexts minP maxP noClickCost clickCost tot_iter mod_iter rnd_seed swap_preferences_iter ml_args2 SaveModelPath");
                return;
            }

            string ml_args2 = args[11];

            string SaveModelPath = "";
            if (args.Length >= 13)
                SaveModelPath = args[12];


            VowpalWabbitSimulator.Run(ml_args, tot_iter, mod_iter, rnd_seed, numContexts, numActions, minP, maxP, noClickCost, clickCost, swap_preferences_iter, ml_args2, SaveModelPath);
        }
    }
}
