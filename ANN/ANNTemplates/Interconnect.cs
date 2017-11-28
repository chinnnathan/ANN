using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class NeuronInterconnect
        ///<summary>The weight+output values connecting each layer</summary>
    {
        private double _update = 0.0d;

        public double Weight { get; set; } //weight of the connection
        public double IValue { get; set; } //input value of the last neuron
        public double FValue { get; set; } //value of the last neuron
        public double DValue { get; set; } //derivative value of neuron
        public double Error { get; set; } //Error value of neuron

        public double Update { get { return _update; } set { _update = value; } } //Amount to Update the layers by

    }

    public static class Interconnect
    {
        public static double[] RunLayerOutputs(Layer outLayer)
        {
            double[] layerOutputs = new double[outLayer.Count];
            int weightLength = 0;
            foreach (var layer in outLayer.NextLayers)
                weightLength += layer.Count;
            double[][] weights = new double[weightLength][];
            for (int i = 0; i < weights.Count(); i++)
            {
                weights[i] = new double[outLayer.Count];
            }
            return layerOutputs;
        }
    }

    public struct DynamicInterconnect
    {
        public double Update { get; set; }
        public List<double> ForwardPass { get; set; }
        public List<double> BackwardPass { get; set; }    
    }

    public struct NeuronInternal
    {
        public double[] Update { get; set; }
        public int Index { get; set; }
        public double SidePass { get; set; }
        public double ForwardPass { get; set; }
        public double BackwardPass { get; set; }
        public List<double> MetaData { get; set; }
        public List<double> Weights { get; set; }
    }
}
