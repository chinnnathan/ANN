using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN.ANNTemplates;
using System.Diagnostics;

namespace ANN.Networks
{
    public class Backpropagation : Network
    {

        public Backpropagation() { }

        public Backpropagation(int outputnodes, int hiddenlayers=1)
        {
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

            for(int i=0; i<hiddenlayers; i++)
            {
                Add(new Layer()
                {   Index = i + 1,
                    Label = string.Format("HiddenLayer_{0}", i),
                    Type = LayerType.Hidden,
                });
            }
            Add(output);
        }

        new public bool Run()
        {
            foreach(var layer in this)
            {
                layer.FeedForward();
            }
            return true;
        }

        new public bool Train()
        {
            foreach (var layer in this.Where(x => x.Index != this.Count))
            {
                layer.FeedForward();
            }
            return true;
        }

    }
}
