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
                    var supd = new List<double>(update);
                    supd.Sort();
                    foreach (var e in supd)
                    {
                        xi = update.IndexOf(e); // input nodes result
                        if (!_ifound.Any(x => x == xi))
                        {
                            //if (Update(update, Inputs[yi], xi))
                            if (Update(Inputs[yi], xi))
                                return;
                        }
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

        private void RunGlobalError()
        {
            var neurons = new List<int>(_ifound.Where(x=>x>=0));
            for (int index = 0; index < Inputs.Count; index++)
            {
                if (_ifound[index] >= 0)
                    continue;

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

                List<double> Errors = new List<double>();
                for (int i = 0; i < this[Count - 1].Count; i++)
                {
                    List<double[]> Outputs = new List<double[]>();
                    for (int n = 0; n < this[Count - 1].Count; n++)
                    {
                        var neuron = this[Count - 1][n];
                        if (n == i)
                            Outputs.Add(Inputs[index].ToArray());
                        else
                            Outputs.Add(neuron.InterIn.Select(x => x.Weight).ToArray());
                    }

                    double error = 0;
                    for (int n = 0; n < this[Count - 1].Count; n++)
                    {
                        var neuron = this[Count - 1][n];
                        if (n == this[Count - 1].Count - 1)
                        {
                            neuron.Error = ActivationFunctions.Chicago(Outputs[n], Outputs[0]);
                        }
                        else
                        {
                            neuron.Error = ActivationFunctions.Chicago(Outputs[n], Outputs[n + 1]);
                        }
                        error += neuron.Error + neuron.Prediction;
                    }
                    Errors.Add(error);
                }
                List<double> se = new List<double>(Errors);
                se.Sort();
                int iin = -1;
                foreach(var e in se)
                {
                    iin = Errors.IndexOf(e);
                    if (neurons.All(x => x != iin))
                    {
                        neurons.Add(iin);
                        break;
                    }
                }
                Update(Inputs[index], iin);

                //Update(Inputs[index], iin);
                /*if (_ifound.Any(x => x == iin))
                    continue;
                else
                    Update(Inputs[index], iin);*/
            }
        }

        private double GetTotalError()
        {
            double error = 0;
            List<double[]> Outputs = new List<double[]>();
            for (int n = 0; n < this[Count - 1].Count; n++)
            {
                Outputs.Add(this[Count-1][n].InterIn.Select(x => x.Weight).ToArray());
            }

            for (int n = 0; n < this[Count - 1].Count; n++)
            {
                var neuron = this[Count - 1][n];
                if (n == this[Count - 1].Count - 1)
                {
                    neuron.Error = ActivationFunctions.Chicago(Outputs[n], Outputs[0]);
                }
                else
                {
                    neuron.Error = ActivationFunctions.Chicago(Outputs[n], Outputs[n + 1]);
                }
                error += neuron.Error + neuron.Prediction;
            }

            return error;
        }

        new public bool Run(int mode = 3)
        {
            double acc = 0;

            bool allfinish = true;

            switch(mode)
            {
                case 1: RunLocalMinimum();
                    break;
                case 2: RunGlobalMinimum();
                    break;
                case 3: RunGlobalError();
                    break;
                default: RunGlobalError();
                    break;
            }

            allfinish = _ifound.All(x => x >= 0); //init at negative values

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
            Epochs = 0;
            int correct = 0;
            Console.WriteLine("\r[{0}] - Start", DateTime.Now);
            do
            {
                Epochs++;
                int mode = (Epochs > 8000) ? 2:3;
                stop = Run(mode);

                if (_ifound.Where(x => x != -1).Count() > correct)
                {
                    correct++;
                    Console.WriteLine("[{0}] Correct: {1} Current Epochs: {2}", DateTime.Now, correct, Epochs);
                    for (int i = 0; i < _ifound.Length; i++)
                        Console.WriteLine("{0}: neuron - {1}", i, _ifound[i]);

                    if (Radius > 0)
                        Radius--;

                }

            } while (!stop);
            Finished = true;
            _error= GetTotalError();
            Console.WriteLine("Total Distance Travelled: {0}", _error);
            return true;
        }

        private double _foundval = 10;
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

            for (int r = -1 * Radius; r <= Radius; r++)
            {
                int newi = ineur + r;
                if (newi >= this[Count - 1].Count)
                    newi = newi - this[Count - 1].Count;
                if (newi < 0)
                    newi = this[Count - 1].Count + newi;

                if (_ifound.Any(x => x == newi))
                    return false;

                var neuron = this[Count - 1][newi];
                
                neuron.Update = min;
                double lr = (newi == ineur) ? neuron.LearningRate : neuron.LearningRate * 0.01/Math.Abs(r);

                Parallel.For(0, neuron.InterIn.Count, i =>
                {
                    neuron.InterIn[i].Weight += lr * (input[i] - neuron.InterIn[i].Weight);
                });

                GetGraph(newi);
            }
            return true;
        }

        public bool Update(List<double> input, int ineur)
        {
            if (this[Count-1][ineur].Prediction <= _foundval)
            {
                _ifound[Inputs.IndexOf(input)] = ineur;
                return false;
            }

            for (int r = -1 * Radius; r <= Radius; r++)
            {
                int newi = ineur + r;
                int inind = Inputs.IndexOf(input);

                if (newi >= this[Count - 1].Count)
                    newi = newi - this[Count - 1].Count;
                if (newi < 0)
                    newi = this[Count - 1].Count + newi;

                if (_ifound.Any(x => x == newi)) //this neuron has been assigned
                    return false;

                if (_ifound[inind] >= 0) //this input point has a corresponding value
                    return false;

                var neuron = this[Count - 1][newi];
                
                double lr = (newi == ineur) ? neuron.LearningRate : neuron.LearningRate * 0.01/Math.Abs(r);

                Parallel.For(0, neuron.InterIn.Count, i =>
                {
                    neuron.InterIn[i].Weight += lr * (input[i] - neuron.InterIn[i].Weight);
                    /*if (newi == ineur)
                        neuron.InterIn[i].Weight = input[i];
                    else
                        neuron.InterIn[i].Weight += lr * (input[i] - neuron.InterIn[i].Weight);//*/
                });

                GetGraph(newi);
            }
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
