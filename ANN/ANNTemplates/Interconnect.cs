﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class NeuronInterconnect
        ///<summary>The weight+output values connecting each layer</summary>
    {
        public double Weight { get; set; } //weight of the connection
        public double IValue { get; set; } //input value of the last neuron
        public double FValue { get; set; } //value of the last neuron
        public double DValue { get; set; } //derivative value of neuron
        public double Error { get; set; } //Error value of neuron
    }

    public static class Interconnect
    {
        public static double[] RunLayerOutputs(Layer outLayer)
        {
            double[] layerOutputs = new double[outLayer.Count];
            double[][] weights = new double[outLayer.NextLayer.Count][];
            for (int i = 0; i < weights.Count(); i++)
            {
                weights[i] = new double[outLayer.Count];
            }
            return layerOutputs;
        }
    }
}
