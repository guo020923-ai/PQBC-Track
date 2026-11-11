"""Unmodified official metric functions, exclusive report/cache directory."""
import os,sys,json,argparse,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from center_joint_contract_20260930 import NAME
from test_center_joint_20260930 import verify_outputs
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.environment import env_settings
from lib.test.analysis import plot_results,extract_results

def run_official_metrics(trackers,dataset,report_name,report_root):
    report_root=Path(report_root);report_root.mkdir(parents=True,exist_ok=False)
    settings=copy.copy(env_settings());settings.result_plot_path=str(report_root)
    # Only redirect report caches in this independent process; metrics unchanged.
    previous=(plot_results.env_settings,extract_results.env_settings)
    plot_results.env_settings=lambda:settings
    extract_results.env_settings=lambda:settings
    try:
        plot_results.print_results(trackers,dataset,report_name=report_name,merge_results=True,
            force_evaluation=True,plot_types=('success','prec','norm_prec'),save_file=str(report_root/'metrics.txt'))
    finally:plot_results.env_settings,extract_results.env_settings=previous

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tnl2k','lasot','otb99'],required=True)
    p.add_argument('--epoch',type=int,default=30);p.add_argument('--runid',type=int,required=True)
    args=p.parse_args();os.chdir(str(ROOT))
    tracker=Tracker('uvltrack',NAME,args.dataset,run_id=args.runid,eval=True,
        display_name='CenterJoint-E'+str(args.epoch))
    manifest=json.loads((Path(tracker.results_dir)/'center_joint_run_manifest.json').read_text())
    if (manifest['config'],manifest['dataset'],manifest['epoch'],manifest['runid'])!=(NAME,args.dataset,args.epoch,args.runid):
        raise ValueError('Wrong dataset/epoch/run identity')
    dataset=get_dataset(args.dataset);count=verify_outputs(dataset,tracker,args.epoch,require_all=True)
    report_name=args.dataset+'_NLBBOX_'+f'{args.epoch:03d}'
    report_root=ROOT/'change/center_joint_evaluation_20260930'/f'{NAME}_{args.dataset}_e{args.epoch:04d}_run{args.runid}'
    run_official_metrics([tracker],dataset,report_name,report_root)
    (report_root/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('EVALUATION_COMPLETE',json.dumps(dict(sequences=count,expected=len(dataset),report=str(report_root))),flush=True)

if __name__=='__main__':main()
