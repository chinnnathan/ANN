using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using ANN.ANNTemplates;
using ANN.Utils;
using System.Diagnostics;

namespace ANN.Networks
{
    public class SOM : Network
    {
        //private ActivationFunctions.Del1 _output = ActivationFunctions.SoftMax;

        public int Radius { get; set; }

        public SOM() { }

        public SOM(int outputnodes, int hiddenlayers = 0)
        {
            _output = ActivationFunctions.Chicago;
            Layer input = new Layer()
            {
                Index = 0,
                Label = "Input",
                Type = LayerType.Input,
            };
            Layer output = new Layer()
            {
                Index = hiddenlayers + 1,
                Label = "Output",
                Type = LayerType.Output,
            };
            Add(input);

            for (int i = 0; i < hiddenlayers; i++)
            {
                Add(new Layer()
                {
                    Index = i + 1,
                    Label = string.Format("HiddenLayer_{0}", i),
                    Type = LayerType.Hidden,
                });
            }
            Add(output);
        }

        static Random rnd = new Random();

        new public bool Run()
        {
            double acc = 0;

            bool allfinish = true;

            for (int index = 0; index < Inputs.Count; index++)
            //int index = rnd.Next(Count);
            //int index = 5;
            {
                Parallel.ForEach(this.Where(x => x.Type == LayerType.Input), layer =>
                {
                    Parallel.For(0, layer.Count, i =>
                    {
                        layer[i].InterIn[0].IValue = Inputs[index][i];
                        layer[i].InterIn[0].DValue = Inputs[index][i];
                        layer[i].InterIn[0].FValue = Inputs[index][i];
                        layer[i].Input = Inputs[index][i];
                    });
                });

                foreach (var layer in this.Where(x => x.Index != this.Count))
                {
                    layer.FeedForward();
                    //acc = GetOutputAccuracy();
                }

                /*Parallel.ForEach(this[Count - 1], 
                  n =>
                  {
                      n.Predictions = new double[] { n.InterIn[0].Weight, n.InterIn[1].Weight };
                  });//*/

                double[] ul = new double[Classes];
                Parallel.For(0, Classes, 
                    i =>
                    {
                        ul[i] = ActivationFunctions.Chicago(Inputs[i].ToArray(), this[Count-1][i].InterIn.Select(x=>x.Weight).ToArray());
                    });


                /*allfinish = allfinish & Update(this[this.Count-1].Select(x=> 
                    ActivationFunctions.Chicago(
                        x.InterIn.Select(y=>y.IValue).ToArray(),
                        x.InterIn.Select(y => y.Weight).ToArray())).ToList(), 
                    Inputs[index]);//*/
                //allfinish = allfinish & Update(this[this.Count - 1].Select(x => x.Prediction).ToList(), Inputs[index], index);
                allfinish = allfinish & Update(ul.ToList(), Inputs[index], index);
            }

            return allfinish;
        }

        public void GetGraph(int index)
        {
            if (Graph.All(x => x.Count == Classes) && Graph.Count > 0)
            {
                Graph[0][index] = new Tuple<double, double>(Inputs[index][0], Inputs[index][1]);
                var output = this[1][index];
                Graph[1][index] = new Tuple<double, double>(output.InterIn[0].Weight, output.InterIn[1].Weight);
            }
            else
            {
                var gr = new Tuple<double, double>[Classes];
                Parallel.For(0, Inputs.Count, i =>
                {
                    gr[i] = new Tuple<double, double>(Inputs[i][0], Inputs[i][1]);
                });
                Graph.Add(gr.ToList());

                gr = new Tuple<double, double>[Classes];
                var outputs = this.Where(l => l.Type == LayerType.Output).SelectMany(x => x).ToList();
                Parallel.ForEach(outputs, output =>
                {
                    int i = outputs.IndexOf(output);
                    gr[i] = new Tuple<double, double>(output.InterIn[0].Weight, output.InterIn[1].Weight);
                });
                Graph.Add(gr.ToList());
            }
        }

        private double _min = 6E17;
        private double[] _mins;

        public bool Train(bool debug)
        {
            _mins = new double[Classes];
            for (int i = 0; i < Classes; i++)
                _mins[i] = _min;

            Finished = false;
            bool stop;
            Console.WriteLine("\r[{0}] - Start", DateTime.Now);
            do
            {
                stop = Run();
            } while (!stop);
            Finished = true;
            return true;
        }

        public bool Update(List<double> list, List<double> input, int ineur)
        {
            var sl = new List<double>(list);
            sl.Sort();
            double foundval = 0.05;
            double min = 0;
            int index = 0;

            for (int i = 0; i < Classes; i++)
            {
                min = sl[i];
                index = list.IndexOf(min);

                if (_mins[index] > foundval)
                {
                    _mins[index] = min;
                    break;
                }
            }

            if (min < _min) _min = min;

            if (min <= foundval)
            {
                Console.WriteLine("\r[{0}] - Zero Value Found", DateTime.Now);
                return true;
            }
            input = Inputs[index];


            var neuron = this[Count - 1][ineur];
            neuron.Update = min;
            double lr = neuron.LearningRate;

            Parallel.For(0, neuron.InterIn.Count, i =>
            {
                neuron.InterIn[i].Weight += lr * (input[i] - neuron.InterIn[i].Weight);
            });

            GetGraph(ineur);
            return false;
        }
    }
}
