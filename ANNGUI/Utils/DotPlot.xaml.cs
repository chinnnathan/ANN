using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Defaults;
using System.Collections.Generic;
using ANNGUI;
using Wpf.CartesianChart;

namespace Wpf.CartesianChart.ScatterPlot
{
    /// <summary>
    /// Interaction logic for ScatterExample.xaml
    /// </summary>
    public partial class ScatterExample : UserControl
    {

        internal static ScatterExample scatter;

        public ScatterExample()
        {
            InitializeComponent();

            var r = new Random();
            ValuesA = new ChartValues<ObservablePoint>();
            ValuesB = new ChartValues<ObservablePoint>();

            int count = ANNGUI.MainWindow.main.GetClasses;

            for (var i = 0; i <= count; i++)
            {
                ValuesA.Add(new ObservablePoint(r.NextDouble() * 10, r.NextDouble() * 10));
                ValuesB.Add(new ObservablePoint(r.NextDouble() * 10, r.NextDouble() * 10));
            }

            scatter = this;
            DataContext = this;
        }

        /*public ScatterExample()
        {
            //var series = ANNGUI.MainWindow.main.GetWeights;

            InitializeComponent();
            ValuesA = new ChartValues<ObservablePoint>();
            ValuesB = new ChartValues<ObservablePoint>();
            if (Series != null)
            {
                foreach (var vals in Series[0])
                {
                    ValuesA.Add(new ObservablePoint(vals.Item1, vals.Item2));
                }
                foreach (var vals in Series[1])
                {
                    ValuesB.Add(new ObservablePoint(vals.Item1, vals.Item2));
                }
            }

            DataContext = this;
        }*/
        
        
        public List<List<Tuple<double,double>>> Series { get; set; }

        public ChartValues<ObservablePoint> ValuesA { get; set; }
        public ChartValues<ObservablePoint> ValuesB { get; set; }
        //public ChartValues<ObservablePoint> ValuesC { get; set; }
        
        internal void GraphNewData()
        {
            if (Series.Count > 0)
            {
                for (int i = 0; i < ValuesA.Count - 1; i++)
                {
                    ValuesA[i].X = Series[0][i].Item1;
                    ValuesA[i].Y = Series[0][i].Item2;
                    ValuesB[i].X = Series[1][i].Item1;
                    ValuesB[i].Y = Series[1][i].Item2;
                }
                ValuesA[ValuesA.Count - 1].X = Series[0][0].Item1;
                ValuesA[ValuesA.Count - 1].Y = Series[0][0].Item2;
                ValuesB[ValuesB.Count - 1].X = Series[1][0].Item1;
                ValuesB[ValuesB.Count - 1].Y = Series[1][0].Item2;
            }
        }

        private void RandomizeOnClick(object sender, RoutedEventArgs e)
        {
            Series = ANNGUI.MainWindow.main.GetWeights;

            if (Series.Count > 0)
            {
                /*ValuesA = new ChartValues<ObservablePoint>();
                ValuesB = new ChartValues<ObservablePoint>();

                foreach (var vals in Series[0])
                {
                    ValuesA.Add(new ObservablePoint(vals.Item1, vals.Item2));
                }
                foreach (var vals in Series[1])
                {
                    ValuesB.Add(new ObservablePoint(vals.Item1, vals.Item2));
                }*/
                for(int i=0; i < ValuesA.Count; i++)
                {
                    ValuesA[i].X = Series[0][i].Item1;
                    ValuesA[i].Y = Series[0][i].Item2;
                    ValuesB[i].X = Series[1][i].Item1;
                    ValuesB[i].Y = Series[1][i].Item2;
                }
            }

            else
            {
                for(int i = 0; i < ValuesA.Count; i++)
                {
                    ValuesA[i].X = 50 * i;
                    ValuesA[i].Y = 125.4 * i;
                }
            }

        }
    }
}