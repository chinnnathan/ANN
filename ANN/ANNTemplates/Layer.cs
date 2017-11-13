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
        public List<Layer> PreviousLayers = new List<Layer>();
        public List<Layer> NextLayers = new List<Layer>();

        private double[,] _weights;
        private double[,] _values;
        //private double[,] _outputs;
        private double[][] _outputs;
        private double[][] _gradients;

        public string Label { get; set; }
        public int Index { get; set; }
        public LayerType Type { get; set; }
        public double[] Predictions { get { return this.Select(x=>x.Prediction).ToArray(); } }
        public double[] OutputWeights { get { return this.SelectMany(x => x.InterOut.Select(y => y.Weight)).ToArray(); } }

        public void FeedForward()
        {
            try
            {
                if (Type == LayerType.Output)
                {
                    List<double> v = this.Select(x => x.Input).ToList();
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.Prediction = (double) neuron.RunActivation(neuron.Input, v.ToArray());
                        neuron.Error = GradientFunctions.SumSquaredError(neuron.Expected, neuron.Prediction);
                        neuron.BpropValue = (double)neuron.RunGradient(neuron.Prediction) * neuron.Error;

                        Parallel.ForEach(neuron.InterIn, input =>
                        {
                            input.Error = neuron.Error;
                        });
                    });
                }

                else
                {
                    Parallel.ForEach(this, neuron =>
                    {
                        double bias = neuron.InterIn[0].Weight;
                        neuron.Prediction = (double)neuron.RunActivation(neuron.Input + bias);
                        neuron.BpropValue = (double)neuron.RunGradient(neuron.Input);
                    });

                    _values = AdvancedMath.ConvertToMatrix(
                        this.Select(x => x.Prediction).ToArray(),
                        -1, 1);

                    _weights = AdvancedMath.ConvertToMatrix(this.SelectMany(
                        x => x.InterOut).Select(w => w.Weight).ToArray(),
                        this[0].InterOut.Count, this.Count);

                    _outputs = AdvancedMath.JMultiplyMatrix(_weights, _values);

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterOut, inter =>
                        {
                            inter.FValue = neuron.Prediction;
                            inter.IValue = neuron.Input;
                        });
                    });

                    var ConnectedNeurons = NextLayers.SelectMany(x => x.Select(n => n)).ToList();
                    Parallel.For(0, ConnectedNeurons.Count, i =>
                    {
                        ConnectedNeurons[i].Input = _outputs[i][0];
                    });

                    /*Parallel.For(0, _outputs.Count(), i =>
                    {
                        Parallel.For(0, _outputs[i].Count(), j =>
                        {
                            NextLayer[i].Input = _outputs[i][j];
                        });
                    });*/
                }

                Parallel.ForEach(this, neuron =>
                {
                    Parallel.ForEach(neuron.InterIn, inter =>
                    {
                        inter.DValue = neuron.BpropValue;
                    });
                });
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
                        Parallel.For(0, this[i].InterIn.Count, j =>
                        {
                            this[i].InterIn[j].Update += this[i].LearningRate * this[i].BpropValue * this[i].Prediction;
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
                    _weights = AdvancedMath.ConvertToMatrix(this.SelectMany(
                        x => x.InterOut).Select(w => w.Weight).ToArray(),
                        Count, this[0].InterOut.Count);

                    _values = AdvancedMath.ConvertToMatrix(this[0].InterOut.Select(
                        x => x.DValue).ToArray(),
                        -1, 1);

                    _outputs = AdvancedMath.JMultiplyMatrix(_weights, _values);

                    Parallel.For(0, Count, i =>
                    {
                        this[i].BpropValue = _outputs[i][0] * (double)this[i].RunGradient(this[i].Input);
                    });

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterIn, inter =>
                        {
                            inter.Update += neuron.LearningRate * neuron.BpropValue * inter.FValue;
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
                            //Weight = AdvancedMath.GetRandomRange(min, max),
                            Weight = 1,
                        };
                        neuron.InterIn.Add(icIn);
                    }

                    

                    foreach (var nextLayer in NextLayers)
                    {
                        if (nextLayer.Type == LayerType.Context)
                        {
                            NeuronInterconnect icl1 = new NeuronInterconnect()
                            {
                                Weight = 1,
                            }; // initialize layer1

                            int i = IndexOf(neuron);
                            neuron.InterOut.Add(icl1);
                            nextLayer[i].InterIn.Add(icl1);
                            neuron.AfterNeurons.Add(nextLayer[i]);
                            nextLayer[i].PriorNeurons.Add(neuron);
                        }

                        else
                        {
                            foreach (var nextNeuron in nextLayer)
                            {
                                NeuronInterconnect icl1 = new NeuronInterconnect()
                                {
                                    Weight = AdvancedMath.GetRandomRange(min, max),
                                }; // initialize layer1
                                NeuronInterconnect icl2 = icl1; // reference layer1 for use in layer2
                                neuron.InterOut.Add(icl1);
                                nextNeuron.InterIn.Add(icl2);
                                neuron.AfterNeurons.Add(nextNeuron);
                                nextNeuron.PriorNeurons.Add(neuron);
                            }
                        }
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
        Context,
        Output,
    }
}
