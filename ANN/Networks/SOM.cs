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


        private void RunLocalMinimum()
        {
            List<double[]> fullupdate = new List<double[]>();
            for (int index = 0; index < Inputs.Count; index++)
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
                    layer.RunMap();
                }

                double[] ul = new double[Classes];
                Parallel.For(0, Classes,
                    i =>
                    {
                        ul[i] = this[Count - 1][i].Prediction;
                    });
                fullupdate.Add(ul);
            }

            List<double[]> upavail = new List<double[]>();
            foreach (var u in fullupdate)
            {
                foreach (int i in _ifound.Where(x => x >= 0 && x < Count))
                    u.ToList().Remove(i);
                upavail.Add(u);
            }

            //List<double> all = fullupdate.SelectMany(x => x.Select(y => y)).Where(x => x > _foundval).ToList();
            List<double> all = upavail.SelectMany(x => x.Select(y => y)).Where(x => x > _foundval).ToList();
            all.Sort();
            int minindex = 0;
            double fullmin = all.Min();
            for (int j = 0; j < all.Count; j++)
            {
                fullmin = all[j];
                for (int i = 0; i < fullupdate.Count; i++)
                {
                    if (fullupdate[i].Any(x => x == fullmin))
                    {
                        minindex = i;
                        //int ineur = fullupdate[i].ToList().IndexOf(fullmin);
                        if (_ifound.Any(x => x == minindex))
                            break;
                        else if (Update(fullupdate[minindex].ToList(), Inputs[minindex], minindex))
                            return;
                    }
                }
            }
        }


        private void RunGlobalMinimum()
        {
            List<double[]> fullupdate = new List<double[]>();
            for (int index = 0; index < Inputs.Count; index++)
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
                    layer.RunMap();
                }

                double[] ul = new double[Classes];
                Parallel.For(0, Classes, 
                    i =>
                    {
                        ul[i] = this[Count-1][i].Prediction;
                    });
                fullupdate.Add(ul);
            }

            List<double> sum = fullupdate.Select(x => x.Sum()).ToList();
            List<double> ssm = new List<double>(sum);
            ssm.Sort();
            int yi, xi;
            for (int i = 0; i < ssm.Count; i++) 
            {
                yi = sum.IndexOf(ssm[i]); // output nodes result
                List<double> update = fullupdate[yi].ToList();
                if (update.Min() > _foundval)
                {
                    xi = update.IndexOf(update.Min()); // input nodes result
                    if (!_ifound.Any(x => x == xi))
                    {
                        if (Update(update, Inputs[yi], xi))
                            break;
                    }
                }
                else
                {
                    xi = update.IndexOf(update.Min()); // input nodes result
                    if (!_ifound.Any(x => x == xi))
                        _ifound[yi] = xi;
                }
            }
        }

        new public bool Run()
        {
            double acc = 0;

            bool allfinish = true;

            //RunLocalMinimum();
            RunGlobalMinimum();

            allfinish = _ifound.All(x => x > 0); //init at negative values

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
        private int[] _ifound;

        public bool Train(bool debug)
        {
            _mins = new double[Classes];
            _ifound = new int[Classes];
            for (int i = 0; i < Classes; i++)
            {
                _mins[i] = _min;
                _ifound[i] = -1;
            }

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

        private double _foundval = 1;
        public bool Update(List<double> list, List<double> input, int ineur)
        {

            var sl = new List<double>(list);
            sl.Sort();
            double min = 0;
            int index = 0;

            for (int i = 0; i < Classes; i++)
            {
                min = sl[i];
                index = list.IndexOf(min);
                if(min <= _foundval)
                {
                    if (_ifound.Any(x => x == ineur))
                        continue;
                    else
                        _ifound[ineur] = index;
                }

                else if (_mins[index] > min)
                {
                    _mins[index] = min;
                    break;
                }
            }

            if (min < _min) _min = min;

            if (min <= _foundval)
            {
                _ifound[ineur] = index;
                Console.WriteLine("\r[{0}] - Zero Value Found", DateTime.Now);
                return false;
            }

            var neuron = this[Count - 1][ineur];
            neuron.Update = min;
            double lr = neuron.LearningRate;

            Parallel.For(0, neuron.InterIn.Count, i =>
            {
                neuron.InterIn[i].Weight += lr * (input[i] - neuron.InterIn[i].Weight);
            });

            GetGraph(ineur);
            return true;
        }

        public bool Update(List<double> list, List<double> input, int ineur, int ix)
        {
            if (_ifound.Any(x => x == ineur))
                return false;

            double foundval = 1;
            double min = list.Min();

            if (_mins[ix] > min)
            {
                _mins[ix] = min;
            }

            if (min < _min) _min = min;

            if (min <= foundval)
            {
                _ifound[ix] = ineur;
                Console.WriteLine("\r[{0}] - Zero Value Found", DateTime.Now);
                return true;
            }

            input = Inputs[ix];
            var neuron = this[Count - 1][ineur];

            double lr = neuron.LearningRate;

            Parallel.For(0, neuron.InterIn.Count, i =>
            {
                neuron.InterIn[i].Weight += lr * (input[i] - neuron.InterIn[i].Weight);
            });

            GetGraph(ineur);
            return true;
        }
    }
}
