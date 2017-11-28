using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using ANN.ANNTemplates;
using ANN.Utils;
using System.Text.RegularExpressions;

namespace ANN.Networks
{
    class SOMDynamic : List<DynamicLayer>
    {
        private bool _debug = true;
        private bool _finished = false;
        private ActivationFunctions.Del2 _output = ActivationFunctions.Distance;
        public ActivationFunctions.Del2 Output { get { return _output; } set { _output = value; } }

        public int Radius = 1;
        public int MapDimensions = 1;

        public List<List<Tuple<double, double>>> Graph { get; set; }

        public List<List<double>> Inputs = new List<List<double>>();
        public List<List<double>> Correct = new List<List<double>>();
        public int Classes { get; set; }
        public int Epochs { get; set; }
        public double LearningRate { get; set; }
        public bool Debug { get { return _debug; } set { value = _debug; } }

        public bool Finished { get { return _finished; } set { _finished = value; } }

        public SOMDynamic() { }

        public SOMDynamic(int outputnodes, int hiddenlayers=0)
        {

            DynamicLayer input = new DynamicLayer(outputnodes)
            {
                Index = 0,
                Label = "Input Layer",
                Type = LayerType.Input,
            };
            Add(input);

            DynamicLayer output = new DynamicLayer(outputnodes)
            {
                Index = hiddenlayers + 1,
                Label = "Output Layer",
                Type = LayerType.Output,
            };

            for (int i=0; i<hiddenlayers; i++)
            {
                Add(new DynamicLayer(outputnodes)
                {
                    Index = i+1,
                    Label = "Hidden Layer",
                    Type = LayerType.Hidden,
                });
            }
            Add(output);
        }

        private int _ignoreCol = 0;
        public void InitializeStd(int[] nodes, Delegate activation, Delegate gradient, double min = -1, double max = 1)
        {

            foreach (var layer in this.Where(l => l.Type == LayerType.Hidden || l.Type == LayerType.Input))
            {
                foreach (var next in this.Where(l => ((l.Type == LayerType.Hidden) || (l.Type == LayerType.Output)) && (l.Index == layer.Index + 1)))
                {
                    layer.NextLayers.Add(next);
                    next.PreviousLayers.Add(layer);
                }

                foreach (var ctxlayer in this.Where(l => l.Type == LayerType.Context && l.Index == layer.Index))
                {
                    layer.NextLayers.Add(ctxlayer);
                    layer.PreviousLayers.Add(ctxlayer);
                    ctxlayer.NextLayers.Add(layer);
                    ctxlayer.PreviousLayers.Add(layer);
                }
            }

            var io = this.Where(l => l.Type == LayerType.Input).SelectMany(x => x)
                .Zip(this.Where(l => l.Type == LayerType.Output).SelectMany(x => x)
                .Zip(Enumerable.Range(0, Classes), Tuple.Create), 
                (i,tup) => new { Input = i, Output = tup.Item1, Index = tup.Item2 });

            try
            {
                var imax = Inputs.SelectMany(x => x).Max();
                var imin = Inputs.SelectMany(x => x).Min();
                var xi = Inputs[6][1];
                var yi = Inputs[6][2];

                Parallel.ForEach(io, neuron =>
                {
                    neuron.Input.Internal = new NeuronInternal()
                    {
                        MetaData = Inputs[neuron.Index],
                        Weights = Inputs[neuron.Index],
                        Update = new double[] { 0, 0 },
                    };
                    neuron.Input.Internal.Weights.RemoveAt(0);

                    neuron.Input.OutInter = new List<DynamicInterconnect>()
                    {
                    new DynamicInterconnect() {ForwardPass = neuron.Input.Internal.Weights},
                    };

                    neuron.Output.Internal = new NeuronInternal()
                    {
                        MetaData = Inputs[neuron.Index],
                        Weights = new List<double>()
                            { /*0,0//*/
                            /*xi + AdvancedMath.GetRandomRange(-200,200),
                            yi + AdvancedMath.GetRandomRange(-200,200),//*/
                            AdvancedMath.GetRandomRange(imin, imax),
                            AdvancedMath.GetRandomRange(imin, imax)//*/
                            },
                    };

                    neuron.Output.InNeuron = new List<DynamicNeuron>() { neuron.Input };
                    neuron.Output.InInter = neuron.Input.OutInter;

                    neuron.Input.Index = neuron.Index;
                    neuron.Output.Index = neuron.Index;
                });
            }
            catch
            {
                Console.WriteLine("UpdateValues errors");
            }
            
        }

        public bool Update()
        {
            if (this[1].Where(y => y.Ignore == false).Count() <= 0.0)
            {
                _finished = true;
            }

            else
            {
                Parallel.ForEach(this.SelectMany(x => x.Where(y => y.Ignore == false)), neuron =>
                    {
                        for (int i = 0; i < neuron.Weights.Count; i++)
                        {
                            neuron.Weights[i] += LearningRate * neuron.Update[i];
                            neuron.Ignore = true;
                            neuron.Update[i] = 0;
                        }
                    });
            }
            return true;
        }


        public bool Run()
        {
            foreach (var layer in this)
            {
                layer.SOMForward(Radius);
            }

            //var gr = new List<Tuple<double, double>>();
            var gr = new Tuple<double, double>[Classes];
            Parallel.ForEach(this.Where(l => l.Type == LayerType.Input).SelectMany(x=>x), input =>
              {
                  gr[input.Index] = new Tuple<double, double>(input.Internal.Weights[0], input.Internal.Weights[1]);
              });

            Graph = new List<List<Tuple<double, double>>>() { gr.ToList() };

            gr = gr = new Tuple<double, double>[Classes];
            Parallel.ForEach(this.Where(l => l.Type == LayerType.Output).SelectMany(x => x), output =>
            {
                gr[output.Index] = new Tuple<double, double>(output.Internal.Weights[0], output.Internal.Weights[1]);
            });

            Graph.Add(gr.ToList());
            return true;
        }


        public bool Train(bool debug=true)
        {
            Console.WriteLine("[{0}] Starting Training -> Epochs: {1} Training Data Sets: {2} Classes:{3}", DateTime.Now, Epochs, Correct.Count, Classes);
            if (debug) // want to minimize the amount of operations per iteration
            {
                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    /*if (epoch >= (Epochs * 3.0) / 4.0)
                        Radius = 1;*/

                    Run();
                    Update();

                    Console.Write("\r[{0}] Epoch {1} - Accuracy: {2:N4}% Correct: {3} Over: {4} Under:{5} Classified:{6}", DateTime.Now, epoch + 1,
                                0,1,2,3,4,5,6);
                }
            }
            /*double acc = 0;
            if (!debug) // want to minimize the amount of operations per iteration
            {
                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    for (int datasetlabel = 0; datasetlabel < Correct.Count; datasetlabel++)
                    {
                        RunBackprop(datasetlabel);
                        UpdateAllWeights();
                    }
                    Console.Write("\r[{0}] Epoch {1} - Accuracy: {2:N4}% Correct: {3} Over: {4} Under:{5} Classified:{6}", DateTime.Now, epoch + 1,
                                GetOutputAccuracy(), _correct, _overPredict, _underPredict, _classified);
                }
            }
            else
            {
                using (var sw = new StreamWriter("outputs_SOM_.csv".AppendTimeStamp()))
                {
                    sw.WriteLine(",Errors_{0},Accuracy,Correct,Classification,OverPrediction,", string.Join(",Errors_", Enumerable.Range(1, Correct.Count()).ToArray()));
                    for (int epoch = 0; epoch < Epochs; epoch++)
                    { 
                        sw.Write("{0},", epoch);
                        for (int datasetlabel = 0; datasetlabel < Correct.Count; datasetlabel++)
                        {
                            RunBackprop(datasetlabel);
                            acc = GetOutputAccuracy();
                            UpdateAllWeights();
                            sw.Write("{0},", Errors.Last());

                         }
                        sw.WriteLine("{0},{1},{2},{3},", acc, _correct, _classified, _overPredict);
                            Console.Write("\r[{0}] Epoch {1} - Accuracy: {2:N4}% Correct: {3} Over: {4} Under:{5} Classified:{6}", DateTime.Now, epoch + 1,
                                acc, _correct, _overPredict, _underPredict, _classified);
                    }
                }
            }*/
            Console.WriteLine("\nTraining Finished");
            return true;
        }

        public void SetCorrect(string filename, int classes = 1)
        {
            Correct = new List<List<double>>();
            using (var sr = new StreamReader(filename))
            {
                while (!sr.EndOfStream)
                {
                    List<double> rv = new List<double>();
                    string pattern = "[a-zA-z]+";
                    string input = sr.ReadLine();
                    string[] istr = Regex.Split(input, pattern);
                    if (istr.Length > 1 || (input.Length <= 1))
                    {
                        for (int i = 0; i < classes; i++)
                            rv.AddRange(istr.Select(x => Double.Parse(x)).ToArray());

                    }
                    else
                    {
                        for (int i = 0; i < classes; i++)
                        {
                            byte[] inb = Encoding.ASCII.GetBytes(input);
                            rv.AddRange(inb.Select(x => (double)x));
                        }
                    }
                    Correct.Add(rv);
                }
            }
        }

        public void SetInputs(string filename, int cluster, bool normalize = true)
        {
            Inputs = new List<List<double>>();
            using (var sr = new StreamReader(filename))
            {
                while (!sr.EndOfStream)
                {
                    List<double> rv = new List<double>();
                    string pattern = "[ |,a-zA-z]+";
                    for (int i = 0; i < cluster; i++)
                    {
                        string input = sr.ReadLine();
                        string[] istr = Regex.Split(input, pattern);
                        if (istr.Length > 1)
                        {
                            rv.AddRange(istr.Select(x => Double.Parse(x)).ToArray());

                        }
                        else
                        {
                            byte[] inb = Encoding.ASCII.GetBytes(input);
                            rv.AddRange(inb.Select(x => (double)x));
                        }
                    }
                    Inputs.Add(rv);
                }
            }

            if (normalize)
            {
                List<double> distinct = Inputs.SelectMany(x => x.Select(y => y).Distinct()).Distinct().ToList();
                Parallel.ForEach(Inputs, input =>
                {
                    for (int i = 0; i < input.Count; i++)
                    {
                        input[i] = distinct.IndexOf(input[i]) + 0.1; //offset so zeroes don't ruin data
                    }
                });
            }
        }
    }
}
