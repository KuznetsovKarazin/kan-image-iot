"""Separate adjustable UTC labels from stream order and monotonic elapsed time."""
import datetime
import hashlib
import json
from pathlib import Path
import time

VERSION='mlp-v2-clock-audit-1'
HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_utc(value):
    result=datetime.datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset()==datetime.timedelta(0),'Invalid UTC timestamp zone')
    return result


def inspect_clock(events,summary=None):
    require(bool(events),'Empty transport trace')
    times=[parse_utc(e['utc']) for e in events]
    rollbacks=[]
    for i in range(1,len(events)):
        if times[i]<times[i-1]:
            rollbacks.append(dict(line_1based=i+1,previous_utc=events[i-1]['utc'],utc=events[i]['utc'],
                                  delta_seconds=(times[i]-times[i-1]).total_seconds(),
                                  previous_event=events[i-1]['event'],event=events[i]['event']))
    marked=any(any(k in e for k in ['monotonic_ns','event_index','clock_schema']) for e in events)
    elapsed=None;method='legacy_file_order_and_protocol_sequence'
    require(marked,'Missing monotonic clock in MLP capture')
    if marked:
        previous=None
        for i,event in enumerate(events):
            require(event.get('clock_schema')=='host-monotonic-v1','Incomplete/unknown monotonic clock schema')
            require(type(event.get('event_index')) is int and event['event_index']==i,'Host event index order')
            now=event.get('monotonic_ns')
            require(type(now) is int and now>=0 and (previous is None or now>=previous),'Host monotonic timestamp order')
            previous=now
        require(events[0].get('logger_version')==VERSION and events[0].get('logger_sha256')==sha(__file__),'Monotonic logger identity')
        elapsed=(events[-1]['monotonic_ns']-events[0]['monotonic_ns'])/1e9
        method='monotonic_ns_and_event_index_and_protocol_sequence'
    envelope=True
    if summary is not None:
        start=parse_utc(summary['start_utc']);end=parse_utc(summary['end_utc'])
        envelope=start<=times[0]<=times[-1]<=end
    anomaly=bool(rollbacks) or not envelope
    return dict(method=method,utc_backward_steps=rollbacks,utc_envelope_valid=envelope,
                utc_is_adjustable=True,utc_order_warning=anomaly,
                host_duration_seconds=elapsed,
                host_duration_basis='monotonic_ns' if marked else 'unavailable_in_legacy_trace',
                utc_endpoint_difference_seconds=(times[-1]-times[0]).total_seconds(),
                utc_endpoint_difference_is_elapsed_time=False,
                device_timing_unchanged=True,
                scope='UTC retained verbatim; raw protocol order and all device timing checks remain mandatory. Legacy traces cannot establish monotonic host duration.')


def install_logger(serial_link_class):
    """Add fields to new host records only; never synthesize old monotonic values."""
    digest=sha(__file__)
    def log(self,event,**fields):
        reserved={'utc','event','monotonic_ns','event_index','clock_schema','logger_version','logger_sha256'}
        require(not reserved.intersection(fields),'Reserved logger field')
        index=getattr(self,'_clock_fix_event_index',0)
        record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,
                    monotonic_ns=time.monotonic_ns(),event_index=index,clock_schema='host-monotonic-v1',**fields)
        if index==0:record.update(logger_version=VERSION,logger_sha256=digest)
        self.trace.write(json.dumps(record)+'\n');self.trace.flush()
        self._clock_fix_event_index=index+1
    serial_link_class.log=log
