using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;

namespace ANN.Utils
{
    public static class OperationCLI
    {
        public static void SetValueByString(this object obj, string str, string setVal)
        {
            try
            {
                PropertyInfo propertyInfo = obj.GetType().GetProperty(str);
                propertyInfo.SetValue(obj, Convert.ChangeType(setVal, propertyInfo.PropertyType), null);
            }
            catch
            {
                Console.WriteLine("Cannot Find Variable: {0}", str);
            }
        }
    }
}
