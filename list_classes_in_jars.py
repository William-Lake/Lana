from argparse import ArgumentParser
import json
from pathlib import Path
from multiprocessing import Pool
import tempfile
from zipfile import ZipFile
import traceback
import string
import time
import random

import pandas as pd
from tqdm import tqdm


MAX_NUM_JARS = 10


class JarProcessor:
    
    @staticmethod
    def create_jar_df(tmp_dir,jar):
        
        df = pd.DataFrame({
            'Classname':[nm.split('/')[-1] for nm in ZipFile(jar).namelist() if nm.endswith('class')],
        })
        
        df['Path'] = jar.absolute().__str__()
        
        return JarProcessor.create_df_path(tmp_dir,df,jar.name + ''.join(random.choices(string.ascii_letters,k=5)))   
            
    @staticmethod
    def yield_jar_dfs(tmp_dir,jars):
        
        for jar in jars:
            
            yield JarProcessor.create_jar_df(tmp_dir,jar)    
            
    @staticmethod
    def create_df_path(tmp_dir,df,name):
        
        df_path = Path(tmp_dir).joinpath(f'{name}.feather')
        
        df.to_feather(df_path)
        
        del df
        
        return df_path          
    
    @staticmethod  
    def process_jars(idx,tmp_dir,jars):
        
        try:
            
            df_paths = []
        
            for df_path in JarProcessor.yield_jar_dfs(tmp_dir,jars):
                
                df_paths.append(df_path)
                    
            df_path = Path(tmp_dir).joinpath(f'{idx}.feather')
            
            df = None
            
            for dfp in df_paths:
                
                df = pd.read_feather(dfp) if df is None else pd.concat([df,pd.read_feather(dfp)],ignore_index=True)
                
                dfp.unlink()
            
            df.to_feather(df_path)
            
            return df_path
        
        except Exception as e:
            
            traceback.print_exc()

def process_futures(futures):
    
    df = None
    
    pbar = tqdm(total=len(futures))
    
    while len(futures) > 0:
        
        futures_to_remove = []
        
        for future in futures:
            
            if future.ready():
                
                pbar.update(1)
                
                futures_to_remove.append(future)
                
                df_path = future.get()
                
                if df_path is not None:
                
                    df = pd.read_feather(df_path) if df is None else pd.concat([df,pd.read_feather(df_path)],ignore_index=True)
                    
                    df.to_feather('class_jar_mappings.feather')
                    
                    df_path.unlink()
                
        for future in futures_to_remove:
            
            futures.remove(future)
            
        if len(futures) > 0:
            
            time.sleep((MAX_NUM_JARS / 10) * 2)
                
    pbar.close()
    
    return df   

def yield_jars(target_dir):
    
    for jar in target_dir.glob('**/*.jar'):
        
        yield jar

def yield_jar_groups(target_dir):
    
    jars = []
    
    idx = 0
    
    for jar in tqdm(yield_jars(target_dir),leave=False):
    
        if jar.name.split('.')[-2] in ['sources','javadoc']: continue
        
        jars.append(jar)
        
        if len(jars) >= MAX_NUM_JARS:
            
            idx += 1
            
            yield idx, jars
            
            jars = []
            
    if jars:
        
        idx += 1
        
        yield idx, jars
            
def gather_args():
    
    arg_parser = ArgumentParser()
    
    arg_parser.add_argument('target_dir',type=Path)
    
    return arg_parser.parse_args()

def main():
    
    args = gather_args()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        print(tmp_dir)
        
        with Pool() as pool:
            
            futures = []
            
            for idx, jars in tqdm(yield_jar_groups(args.target_dir),leave=False):
                
                futures.append(pool.apply_async(JarProcessor.process_jars,(idx,tmp_dir,jars))) 
    
            df = process_futures(futures)
            
            df.sort_values(['Classname','Path']).reset_index(drop=True).to_feather('class_jar_mappings.feather')    

if __name__ == '__main__':
    
    start_time = time.time()
    
    main()
    
    print(f'It took {time.time() - start_time} seconds to complete.')
    
    