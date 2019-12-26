using Newtonsoft.Json;
using System;
using System.IO;
using System.Linq;
using System.Text;
using VW;
using VW.Labels;

namespace simulator
{
    public static class VowpalWabbitSimulator
    {
        public class SimulatorExample
        {
            private readonly int length;

            private readonly byte[] exampleBuffer;

            public float[] PDF { get; }

            public int preferredAction {get; }

            public SimulatorExample(int numActions, int sharedContext, int preferredAction, float minP, float maxP)
            {
                // ensure preferredAction is between 0 and numActions-1
                this.preferredAction = preferredAction % numActions;
                if (this.preferredAction < 0)
                    this.preferredAction += numActions;

                // generate per user context and PDF with maxP for preferred action and minP for all other actions
                this.PDF = Enumerable.Range(0, numActions).Select(_ => minP).ToArray();
                this.PDF[this.preferredAction] = maxP;

                this.exampleBuffer = new byte[32 * 1024];

                var str = JsonConvert.SerializeObject(
                    new
                    {
                        Version = "1",
                        EventId = "1", // can be ignored
                        a = Enumerable.Range(1, numActions).ToArray(),
                        c = new
                        {
                            // shared user context
                            U = new { C = sharedContext.ToString() },
                            _multi = Enumerable
                                .Range(0, numActions)
                                .Select(i => new { A = new { Constant = 1, Id = i.ToString() }, B = new { Id = i.ToString() } })
                                .ToArray()
                        },
                        p = Enumerable.Range(0, numActions).Select(i => 0.0f).ToArray()
                    });

                // allow for \0 at the end
                this.length = Encoding.UTF8.GetBytes(str, 0, str.Length, exampleBuffer, 0);
                exampleBuffer[this.length] = 0;
                this.length++;
            }

            public VowpalWabbitMultiLineExampleCollection CreateExample(VowpalWabbit vw)
            {
                VowpalWabbitDecisionServiceInteractionHeader header;
                var examples = vw.ParseDecisionServiceJson(this.exampleBuffer, 0, this.length, true, out header);

                var adf = new VowpalWabbitExample[examples.Count - 1];
                examples.CopyTo(1, adf, 0, examples.Count - 1);

                return new VowpalWabbitMultiLineExampleCollection(vw, examples[0], adf);
            }
        }

        private static void ExportScoringModel(VowpalWabbit learner, ref VowpalWabbit scorer)
        {
            scorer?.Dispose();
            using (var memStream = new MemoryStream())
            {
                learner.SaveModel(memStream);

                memStream.Seek(0, SeekOrigin.Begin);

                // Note: the learner doesn't use save-resume as done online
                scorer = new VowpalWabbit(new VowpalWabbitSettings { Arguments = "--quiet", ModelStream = memStream });
            }
        }

        public static void Run(string ml_args, int tot_iter, int mod_iter, int rnd_seed=0, int numContexts=10, int numActions=10, float minP=0.03f, float maxP=0.04f, float noClickCost = 0.0f, float clickCost = -1.0f, int swap_preferences_iter=-1)
        {
            // byte buffer outside so one can change the example and keep the memory around
            var exampleBuffer = new byte[32 * 1024];

            var randGen = new Random(rnd_seed);

            var simExamples = Enumerable.Range(0, numContexts)
                .Select(i => new SimulatorExample(numActions, i, i, minP, maxP))
                .ToArray();

            var scorerPdf = new float[numActions];
            int clicks = 0;
            int goodActions = 0;
            int goodActionsSinceLast = 0;
            float cost;

            using (var learner = new VowpalWabbit(ml_args + " --quiet"))
            {
                for (int iter = 1; iter <= tot_iter; iter++)
                {
                    if (iter == swap_preferences_iter)
                    {
                        simExamples = Enumerable.Range(0, numContexts)
                            .Select(i => new SimulatorExample(numActions, i, -i-1, minP, maxP))
                            .ToArray();
                    }

                    // sample uniformly among contexts
                    int contextIndex = randGen.Next(simExamples.Length);
                    var simExample = simExamples[contextIndex];
                    var costPdf = simExample.PDF;

                    using (var ex = simExample.CreateExample(learner))
                    {
                        var scores = ex.Predict(VowpalWabbitPredictionType.ActionProbabilities, learner);

                        var total = 0.0;
                        foreach (var actionScore in scores)
                        {
                            total += actionScore.Score;
                            scorerPdf[actionScore.Action] = actionScore.Score;
                        }

                        var draw = randGen.NextDouble() * total;
                        var sum = 0.0;
                        uint topAction = 0;
                        foreach (var actionScore in scores)
                        {
                            sum += actionScore.Score;
                            if(sum > draw)
                            {
                                topAction = actionScore.Action;
                                break;
                            }
                        }

                        if (topAction == simExample.preferredAction)
                        {
                            goodActions += 1;
                            goodActionsSinceLast += 1;
                        }
                        
                        // simulate click/noClick behavior
                        if (randGen.NextDouble() < costPdf[topAction])
                        {
                            cost = clickCost;
                            clicks += 1;
                        }
                        else
                            cost = noClickCost;

                        ex.Examples[topAction].Label = new ContextualBanditLabel(topAction, cost, scorerPdf[topAction]);

                        // invoke learning
                        var oneStepAheadScores = ex.Learn(VowpalWabbitPredictionType.ActionProbabilities, learner);

                        if (iter % mod_iter == 0 || iter == tot_iter)
                        {
                            Console.WriteLine("{0},{1},{2},{3}", iter, clicks/(float)iter, goodActions, goodActionsSinceLast);

                            goodActionsSinceLast = 0;
                        }
                    }
                }
            }
        }
    }
}
