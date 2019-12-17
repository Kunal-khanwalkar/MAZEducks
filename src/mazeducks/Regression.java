/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mazeducks;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
/**
 *
 * @author Kunal
 */
public class Regression {
    
    /**
     *
     * @param csize
     * @param ctime
     * @param tuntog
     * @return 
     * @throws java.lang.Exception
     */
    public int MLmodel(int csize, int ctime, int tuntog) throws Exception
    {
        Instances dataset = null;
        DataSource source;
        try
        {
            if (tuntog == 0)
                source = new DataSource(".\\src\\mazeducks\\dataset_normal.arff");
            else
                source = new DataSource(".\\src\\mazeducks\\dataset_tunnel.arff");
            dataset = source.getDataSet();
        } catch (Exception e) {
            System.out.println("Invalid Dataset");
            System.exit(0);
        }
        
         //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);

        //Build model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(dataset);
        //output model
        //System.out.println("LR FORMULA : "+model);	

        // Now Predicting the cost 
        Instance fsize = dataset.lastInstance();    
        Instance ins = new Instance(3);
        ins.setValue(0, csize);
        ins.setValue(1, ctime);
        ins.setValue(2, 0);
        ins.setDataset(dataset);
        
        
        double size = model.classifyInstance(ins);
        size = (int)size;
        //System.out.println("-------------------------");	
        if(size<10)
            size = 10;
        else if (size > 40)
            size = 40;
        return (int) size;
        
    }
    
}
