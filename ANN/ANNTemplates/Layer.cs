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
        private double[][] _gradients;

        public string Label { get; set; }
        public int Index { get; set; }
        public LayerType Type { get; set; }
        public double[] Predictions { get { return this.Select(x=>x.Prediction).ToArray(); } }
        public double[] OutputWeights { get { return _weights.Cast<double>().ToArray(); } }

        public void FeedForward()
        {
            try
            {
                _values = AdvancedMath.ConvertToMatrix(
                    this.Select(x => x.Prediction).ToArray(),
                    -1, 1);

                _weights = AdvancedMath.ConvertToMatrix(
                    this.SelectMany(x => x.InterOut.Select(w => w.Weight)).ToArray(),
                    this[0].InterOut.Count, Count);
                

                if (Type == LayerType.Output)
                {
                    List<double> v = _values.Cast<double>().ToList();

                    //_outputs = new double[_values.Length, 1];
                    _outputs = new double[_values.Length][];
                    Parallel.For(0, _outputs.Length, i =>
                    {
                        _outputs[i] = new double[] { (double)this[i].RunActivation(_values[i, 0], v.ToArray()) };
                        this[i].Prediction = _outputs[i][0];
                        this[i].BpropValue = (double)this[i].RunGradient(this[i].Expected, this[i].Prediction);
                        this[i].Error = GradientFunctions.SumSquaredError(this[i].Expected, this[i].Prediction);
                        /*Parallel.For(0, _outputs[i].Length, j =>
                        {
                            _outputs[i][j] = (double)this[i].RunActivation(_values[i, j], v.ToArray());
                        });*/
                    });

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterIn, input =>
                        {
                            input.Error = neuron.BpropValue;
                        });
                    });
                }
                else
                {
                    _outputs = AdvancedMath.JMultiplyMatrix(_weights, _values);

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterOut, output =>
                        {
                            output.IValue = neuron.Prediction;
                        });
                    });

                    Parallel.For(0, _outputs.Length, i =>
                    {
                        Parallel.For(0, _outputs[i].Length, j =>
                        {
                            _outputs[i][j] = (double)this[i].RunActivation(_outputs[i][j]);
                            this[i].Prediction = _outputs[i][j];
                            this[i].BpropValue = (double)this[i].RunGradient(_outputs[i][j]);
                        });
                        Parallel.For(0, this[i].InterOut.Count, j =>
                        {
                            this[i].InterOut[j].FValue = _outputs[i][0];
                            this[i].InterOut[j].DValue = this[i].BpropValue;
                        });
                       // _outputs[i][0] = (double)this[i].RunActivation(_outputs[i][0]);
                    });

                    /*Parallel.For(0, NextLayer.Count, i =>
                    {
                        NextLayer[i].Input = _outputs[i];
                    });*/
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
                // All Derivative functions are ran in Feedforward for ease
                // Error still needed to be found
                // Update the weights based on backpropagation value
                if (Type == LayerType.Output)
                {
                    //Delta = alpha * (tk-yk) * f'(y_ink)
                    //      = LearningRate * Error Value * Derivative Value
                    Parallel.For(0, Count, i =>
                    {
                        this[i].Update = this[i].LearningRate *
                                this[i].BpropValue * this[i].InterIn[0].DValue; // print out value

                        Parallel.For(0, this[i].InterIn.Count, j =>
                        {
                            this[i].InterIn[j].Weight += this[i].LearningRate *
                                this[i].BpropValue * this[i].InterIn[j].DValue;
                        });
                    });
                }

                else
                {
                    //Delta = Sigma((tk-yk)(f'(yin_k) * wjk)
                    //      where z = this layer, y = next layer
                    //Gradient = alpha * Delta * f'(z_inj) * xi
                    //      = LearningRate * Sum of( Error Value * weight)
                    //          * Derivative Value * Value input
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.BpropValue = 0;
                        neuron.Error = 0;
                        Parallel.ForEach(neuron.InterOut, output =>
                        {
                            neuron.Error += output.Error;
                            neuron.BpropValue += output.Error * output.DValue * output.Weight;
                        });
                    });

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterIn, input =>
                        {
                            input.Error = neuron.Error;
                        });
                    });

                    Parallel.For(0, Count, i =>
                    {
                        this[i].Update = this[i].LearningRate *
                                this[i].BpropValue * this[i].InterIn[0].DValue *
                                this[i].InterIn[0].IValue; // Print out

                        Parallel.For(0, this[i].InterIn.Count, j =>
                        {
                            this[i].InterIn[j].Weight += this[i].LearningRate *
                                this[i].BpropValue * this[i].InterIn[j].DValue *
                                this[i].InterIn[j].IValue;
                        });
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
