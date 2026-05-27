from tbparse import SummaryReader
reader = SummaryReader('experiments/TextIF_full_recon_2_20260506-213402/log')
df = reader.scalars
for tag in df['tag'].unique():
    sub = df[df['tag']==tag].sort_values('step')
    vals = sub['value'].values
    steps = sub['step'].values
    mid = len(vals)//2
    avg1, avg2 = vals[:mid].mean(), vals[mid:].mean()
    trend = 'down' if avg2 < avg1 else 'up'
    print(f'{tag}: epochs {steps[0]}->{steps[-1]} | min={vals.min():.4f} max={vals.max():.4f} | {trend} ({avg1:.4f}->{avg2:.4f})')
