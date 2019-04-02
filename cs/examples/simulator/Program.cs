namespace simulator
{
    class Program
    { 
        private static readonly string help_string = "usage: simulator ml_args num_actions minP maxP baseCost pStrategy tot_iter mod_iter rnd_seed";

        static void Main(string[] args)
        {
            string ml_args = args[0];

            int tot_iter;
            int mod_iter;
            int rnd_seed;
            int num_actions;
            float minP;
            float maxP;
            float base_cost;
            int p_strategy;

            if (!int.TryParse(args[1], out num_actions))
                return;
            if (!float.TryParse(args[2], out minP))
                return;
            if (!float.TryParse(args[3], out maxP))
                return;
            if (!float.TryParse(args[4], out base_cost))
                return;
            if (!int.TryParse(args[5], out p_strategy))
                return;
            if (!int.TryParse(args[6], out tot_iter))
                return;
            if (!int.TryParse(args[7], out mod_iter))
                return;
            if (!int.TryParse(args[8], out rnd_seed))
                return;

            int num_contexts = num_actions;

            VowpalWabbitSimulator.Run(ml_args, tot_iter, mod_iter, rnd_seed, num_contexts, num_actions, minP, maxP, base_cost, p_strategy);
        }
    }
}
