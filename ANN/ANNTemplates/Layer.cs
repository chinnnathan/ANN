using System;
using System.Collections.Generic;
using System.Diagnostics;
using ANN.Utils;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class Layer : List<Neuron>
    {
        // each layer could go to n other layers
        public Layer PreviousLayer;
        public Layer NextLayer;

        private double[,] _weights;
        private double[,] _values;
        //private double[,] _outputs;
        private double[][] _outputs;

        public string Label { get; set; }
        public int Index { get; set; }
        public LayerType Type { get; set; }

        public void FeedForward()
        {
            try
            {
                if (Type == LayerType.Input)
                {
                    _values = AdvancedMath.ConvertToMatrix(
                        this[0].Input.Select(x => x).ToArray(),
                        NextLayer.Count, this[0].Input.Count() / NextLayer.Count);

                    _weights = AdvancedMath.ConvertToMatrix(
                        this.SelectMany(x => x.InterIn.Select(w => w.Weight)).ToArray(),
                        -1, 1);
                }
                else
                {
                    /*_values = AdvancedMath.ConvertToMatrix(
                        this.Select(x => x.Input.Sum()).ToArray(),
                        -1, 1);*/

                    _values = AdvancedMath.ConvertToMatrix(
                        this[0].InterIn.Select(x => x.Value).ToArray(),
                        -1, 1);

                    _weights = AdvancedMath.ConvertToMatrix(
                       this.SelectMany(x => x.InterIn.Select(w => w.Weight)).ToArray(),
                       Count, this[0].InterIn.Count);

                    /*_weights = AdvancedMath.ConvertToMatrix(
                        this.SelectMany(x => x.InterOut.Select(w => w.Weight)).ToArray(),
                        this[0].InterOut.Count, Count);*/
                }

                if (Type == LayerType.Output)
                {
                    List<double> v = _values.Cast<double>().ToList();

                    //_outputs = new double[_values.Length, 1];
                    _outputs = new double[_values.Length][];
                    Parallel.For(0, _outputs.Length, i =>
                    {
                        _outputs[i] = new double[] { (double)this[i].RunActivation(_values[i, 0], v.ToArray()) };
                        /*Parallel.For(0, _outputs[i].Length, j =>
                        {
                            _outputs[i][j] = (double)this[i].RunActivation(_values[i, j], v.ToArray());
                        });*/
                    });
                }
                else
                {
                    if (Type == LayerType.Input)
                        _outputs = AdvancedMath.JMultiplyMatrix(_values, _weights);
                    else
                        _outputs = AdvancedMath.JMultiplyMatrix(_weights, _values);

                    Parallel.For(0, _outputs.Length, i =>
                    {
                        Parallel.For(0, _outputs[i].Length, j =>
                        {
                            _outputs[i][j] = (double)this[i].RunActivation(_outputs[i][j]);
                        });
                        Parallel.For(0, this[i].InterOut.Count, j =>
                        {
                            this[i].InterOut[j].Value = _outputs[i][0];
                        });
                       // _outputs[i][0] = (double)this[i].RunActivation(_outputs[i][0]);
                    });

                    Parallel.For(0, NextLayer.Count, i =>
                    {
                        NextLayer[i].Input = _outputs[i];
                    });
                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }

        public void BackPropagate()
        {
            try
            {
                _weights = AdvancedMath.ConvertToMatrix(
                    this.SelectMany(x => x.InterOut.Select(w => w.Weight)).ToArray(),
                    this[0].InterOut.Count, Count);

                if (Type == LayerType.Input)
                    _values = AdvancedMath.ConvertToMatrix(
                        this[0].Input.Select(x => x).ToArray(),
                        Count, this[0].Input.Count() / Count);
                else
                    _values = AdvancedMath.ConvertToMatrix(
                        this.Select(x => x.Input.Sum()).ToArray(),
                        -1, 1);
                /*_values = AdvancedMath.ConvertToMatrix(
                    this.Select(x => x.Prediction).ToArray(),
                    -1, 1);//*/

                if (Type == LayerType.Output)
                {
                    List<double> v = _values.Cast<double>().ToList();

                    //_outputs = new double[_values.Length, 1];
                    _outputs = new double[_values.Length][];
                    Parallel.For(0, _outputs.Length, i =>
                    {
                        _outputs[i][0] = (double)this[i].RunActivation(_values[i, 0], v.ToArray());
                        /*Parallel.For(0, _outputs[i].Length, j =>
                        {
                            _outputs[i][j] = (double)this[i].RunActivation(_values[i, j], v.ToArray());
                        });*/
                    });
                }

                else
                {
                    _outputs = AdvancedMath.JMultiplyMatrix(_weights, _values);
                    Parallel.For(0, _outputs.Length, i =>
                    {
                        Parallel.For(0, _outputs[i].Length, j =>
                        {
                            _outputs[i][j] = (double)this[i].RunActivation(_values[i, j]);
                        });
                        NextLayer[i].Input = _outputs[i];
                    });
                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }

        public void StandardConnectNeurons(double min, double max)
        {
            try
            {
                foreach (var neuron in this)
                {
                    if (Type == LayerType.Input)
                    {
                        NeuronInterconnect icIn = new NeuronInterconnect()
                        {
                            Weight = AdvancedMath.GetRandomRange(min, max),
                        };
                        neuron.InterIn.Add(icIn);
                    }

                    foreach (var nextNeuron in NextLayer)
                    {
                        NeuronInterconnect icl1 = new NeuronInterconnect()
                        {
                            Weight = AdvancedMath.GetRandomRange(min,max),
                        }; // initialize layer1
                        NeuronInterconnect icl2 = icl1; // reference layer1 for use in layer2
                        neuron.InterOut.Add(icl1);
                        nextNeuron.InterIn.Add(icl2);
                        neuron.AfterNeurons.Add(nextNeuron);
                        nextNeuron.PriorNeurons.Add(neuron);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }
    }

    public enum LayerType
    {
        Input,
        Hidden,
        Output,
    }
}
