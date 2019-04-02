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

            public SimulatorExample(int numActions, int sharedContext, float minP, float maxP)
            {
                // generate distinct per user context with 1 prefered action
                this.PDF = Enumerable.Range(0, numActions).Select(_ => minP).ToArray();
                this.PDF[sharedContext] = maxP;

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
                                // .Select(i => new Dictionary<int, Dictionary<string, int>> { { 0, new Dictionary<string, int> { { i.ToString(), 1 } } } }) // new { A = i.ToString() })
                                .Select(i => new { A = new { Constant = 1, Id = i.ToString() }, B = new { Id = i.ToString() }, X = new { Constant = 1, Id = (numActions*sharedContext + i).ToString() } })
                                .ToArray()
                        },
                        p = Enumerable.Range(0, numActions).Select(i => 0.0f).ToArray()
                    });

                //Console.WriteLine(str);
                //Environment.Exit(0);

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

        public static void Run(string ml_args, int tot_iter, int mod_iter, int rnd_seed, int numContexts, int numActions, float minP, float maxP, float baseCost, int pStrategy)
        {
            // byte buffer outside so one can change the example and keep the memory around
            var exampleBuffer = new byte[32 * 1024];

            var randGen = new Random(rnd_seed);

            var simExamples = Enumerable.Range(0, numContexts)
                .Select(i => new SimulatorExample(numActions, i, minP, maxP))
                .ToArray();

            var scorerPdf = new float[numActions];
            //var histPred = new int[numActions, numContexts];
            //var histPred2 = new int[numActions, numContexts];
            //var histActions = new int[numActions, numContexts];
            //var histCost = new int[numActions, numContexts];
            //var histContext = new int[numContexts];
            int clicks = 0;
            int goodActions = 0;
            int goodActionsSinceLast = 0;

            using (var learner = new VowpalWabbit(ml_args + " --quiet"))
            {
                for (int i = 1; i <= tot_iter; i++)
                {
                    // sample uniform among users
                    int userIndex = randGen.Next(simExamples.Length);
                    var simExample = simExamples[userIndex];
                    var pdf = simExample.PDF;

                    //histContext[userIndex]++;

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

                        int modelAction = (int)scores[0].Action; // scorerPdf.ToList().IndexOf(scorerPdf.Max());
                        //histPred[modelAction, userIndex] += 1;
                        //histActions[topAction, userIndex] += 1;
                        if (topAction == userIndex)
                        {
                            goodActions += 1;
                            goodActionsSinceLast += 1;
                        }

                        //Console.Out.WriteLine($"{topAction} {modelAction} {string.Join(",", scorerPdf)}");

                        // simulate behavior
                        float cost = baseCost;
                        if (randGen.NextDouble() < pdf[topAction])
                        {
                            cost -= 1;
                            //histCost[topAction, userIndex] += 1;
                            clicks += 1;
                            //Console.WriteLine($"iter: {i} topAction: {topAction} p:{scorerPdf[topAction]}");
                        }

                        float pReported = scorerPdf[topAction];
                        switch (pStrategy)
                        {
                            case 1:
                                pReported = 1.0f / numActions;
                                break;
                            case 2:
                                pReported = Math.Max(pReported, 0.5f);
                                break;
                            case 6:
                                pReported = Math.Max(pReported, 0.9f);
                                break;
                            case 7:
                                pReported = 0.9f;
                                break;
                            case 13:
                                pReported = 0.5f;
                                break;
                        }

                        ex.Examples[topAction].Label = new ContextualBanditLabel((uint)topAction, cost, pReported);

                        // invoke learning
                        var oneStepAheadScores = ex.Learn(VowpalWabbitPredictionType.ActionProbabilities, learner);
                        //histPred2[oneStepAheadScores[0].Action, userIndex] += 1;

                        if (i % mod_iter == 0 || i == tot_iter)
                        {
                            //Console.WriteLine($"Iter {i}");
                            //Console.WriteLine($"Clicks {clicks}");
                            //Console.WriteLine($"CTR {clicks / (float)i}");
                            //Console.WriteLine("Hists:");
                            //for (int j = 0; j < pdf.Length; j++)
                            //    Console.WriteLine(j + ": " + histActions[j] + " - " + histPred[j] + "/" + histPred2[j] + " - " + histCost[j] + " - " + pdf[j] + " - " + (histCost[j] / (float)histActions[j]));
                            //Console.WriteLine();

                            //foreach (var currEx in ex.Examples)
                            //{
                            //    var ns = currEx.First();
                            //    //foreach (var ns in currEx)
                            //    {
                            //        //Console.Write($"{ns.Index} ");

                            //        //foreach (var fs in ns)
                            //        {
                            //            var fs = ns.First();
                            //            Console.Write($"{fs.Weight} ");
                            //            //Console.WriteLine($"{fs.FeatureIndex}:{fs.X}:{fs.WeightIndex}:{fs.Weight}");
                            //        }
                            //    }
                            //}
                            //Console.Write($"{ex.Examples.First().Last().First().Weight} ");
                            //Console.WriteLine();

                            Console.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8}", ml_args, numActions, baseCost, pStrategy, rnd_seed, i, clicks/(float)i, goodActions, goodActionsSinceLast);

                            //Console.Out.WriteLine($"{userIndex} {topAction} {modelAction} {string.Join(",", scorerPdf)}");

                            goodActionsSinceLast = 0;
                        }
                    }
                }
            }
        }
    }
}
