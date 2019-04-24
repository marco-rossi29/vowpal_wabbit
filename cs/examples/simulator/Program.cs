namespace simulator
{
    class Program
    { 
        private static readonly string help_string = "usage: simulator ml_args num_actions minP maxP baseCost pStrategy tot_iter mod_iter rnd_seed";

        static void Main(string[] args)
        {
            string ml_args = args[0];

            int tot_iter, mod_iter;
            int rnd_seed;
            int num_actions, num_contexts;
            float minP, maxP;
            float base_cost, delta_cost;
            int p_strategy;

            if (!int.TryParse(args[1], out num_actions))
                return;
            if (!int.TryParse(args[2], out num_contexts))
                return;
            if (!float.TryParse(args[3], out minP))
                return;
            if (!float.TryParse(args[4], out maxP))
                return;
            if (!float.TryParse(args[5], out base_cost))
                return;
            if (!float.TryParse(args[6], out delta_cost))
                return;
            if (!int.TryParse(args[7], out p_strategy))
                return;
            if (!int.TryParse(args[8], out tot_iter))
                return;
            if (!int.TryParse(args[9], out mod_iter))
                return;
            if (!int.TryParse(args[10], out rnd_seed))
                return;

            VowpalWabbitSimulator.Run(ml_args, tot_iter, mod_iter, rnd_seed, num_contexts, num_actions, minP, maxP, base_cost, delta_cost, p_strategy);
        }
    }
}
