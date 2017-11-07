using System;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.ComponentModel;

namespace ANN.Utils
{
    public static class FileManager
    {

        public static string AppendTimeStamp(this string fileName)
        {
            return string.Concat(
            Path.GetFileNameWithoutExtension(fileName),
            System.DateTime.Now.ToString("yyyyMMddHHmmssfff"),
            Path.GetExtension(fileName)
            );
        }

        public static string GetJSONString(this object o)
        {
            var jsonString = JsonConvert.SerializeObject(o);
            return jsonString;
        }

        public static void PrintJSONString(object o)
        {
            var jsonString = JsonConvert.SerializeObject(
                o, Formatting.Indented,
                new JsonConverter[] { new StringEnumConverter() });
            System.Console.WriteLine("{0}", jsonString);
        }

        public static string PrintCSVHeader(this object obj)
        {
            string values = "";
            foreach (PropertyDescriptor des in TypeDescriptor.GetProperties(obj))
            {
                try
                {
                    dynamic val = des.GetValue(obj);
                    if (val is string)
                        throw new Exception("Do not iterate strings");
                    int iterindex = 0, iindex = 0, jindex=0;
                    foreach (var iter in val)
                    {
                        try
                        {
                            foreach (var i in iter)
                            {
                                try
                                {
                                    foreach (var j in i)
                                    {
                                        values += string.Format("{0}[{1}][{2}],", des.Name, iterindex, iindex, jindex);
                                        jindex++;
                                    }
                                }
                                catch
                                {
                                    values += string.Format("{0}[{1}][{2}],", des.Name, iterindex, iindex);
                                }
                                iindex++;
                                jindex = 0;
                            }

                        }
                        catch
                        {
                            values += string.Format("{0}[{1}],", des.Name, iterindex);
                        }
                        iterindex++;
                        iindex = 0;
                    }
                }
                catch
                {
                    values += string.Format("{0},", des.Name);
                }
            }
            return values;
        }

        public static string RecursivePrintElement(object obj)
        {
            string value = "";
            if (obj is string)
                throw new Exception("No String Iteration");
            try
            {
                foreach (var i in (dynamic)obj)
                    value += string.Format("{0},", RecursivePrintElement(i));
            }
            catch
            {
                 value += string.Format("{0}", obj);
            }
                
            return value;
        }

        public static string PrintCSVLine(this object obj)
        {
            string values = "";
            foreach (PropertyDescriptor des in TypeDescriptor.GetProperties(obj))
            {
                try
                {
                    dynamic val = des.GetValue(obj);
                    values += string.Format("{0},", RecursivePrintElement(val));
                    /*dynamic val = des.GetValue(obj);
                    if (val is string)
                        throw new Exception("Do not iterate strings");
                    foreach (var iter in val)
                    {
                        try
                        {
                            foreach (var i in iter)
                            {
                                try
                                {
                                    foreach (var j in i)
                                        values += string.Format("{0},", j);
                                }
                                catch
                                {
                                    values += string.Format("{0},", i);

                                }
                            }
                        }
                        catch
                        {
                            values += string.Format("{0},", iter);
                        }

                    }*/
                }
                catch
                {
                    values += string.Format("{0},", des.GetValue(obj));
                }
            }
            return values;
        }


    }
}
