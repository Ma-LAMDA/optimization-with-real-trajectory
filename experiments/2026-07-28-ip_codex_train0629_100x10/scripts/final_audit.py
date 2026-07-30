#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
REPORT = EXPERIMENT / 'results' / 'report'
RUNS = EXPERIMENT / 'results' / 'runs'
DATASET = REPO / 'data' / 'simulation' / 'train_0629.jsonl'
EXPECTED_SOURCE_SHA256 = '79f961a2ce788fa2219e8ee5343b7fa87ca8d79ed3f3dec6049dca0ff7514ad9'
MODEL = 'gpt-5.6-sol'
RESULT_RE = re.compile(r'<result>\s*([\s\S]*?)\s*</result>')


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result(text: str) -> list[str] | None:
    matches = RESULT_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return sorted(set(value))


def independent_correct(prediction: list[str] | None, answer_text: str) -> bool:
    if prediction is None:
        return False
    answer = json.loads(answer_text)
    if (
        isinstance(answer, list)
        and answer
        and all(isinstance(option, list) for option in answer)
        and all(all(isinstance(item, str) for item in option) for option in answer)
    ):
        return any(prediction == sorted(set(option)) for option in answer)
    if not isinstance(answer, list) or any(not isinstance(item, str) for item in answer):
        raise TypeError('source answer is not a supported fault set')
    return prediction == sorted(set(answer))


def relative(path: Path) -> str:
    return path.relative_to(EXPERIMENT).as_posix()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    temporary.replace(path)


def main() -> int:
    state = load(REPORT / 'state.json')
    accepted_index = load(REPORT / 'accepted_index.json')
    summary = load(REPORT / 'summary.json')
    source_lines = [line for line in DATASET.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    records = {index: json.loads(line) for index, line in enumerate(source_lines, start=1)}
    attempt_dirs = sorted(RUNS.glob('q*_r*/attempt_*'))

    missing_events: list[str] = []
    invalid_events: list[dict[str, Any]] = []
    stdout_event_mismatches: list[str] = []
    missing_metadata: list[str] = []
    wrong_models: list[str] = []
    non_ephemeral_calls: list[str] = []
    unsafe_workdirs: list[str] = []
    unsafe_source_records: list[str] = []
    unsafe_prompts: list[str] = []
    allowed_hook_violations: list[str] = []
    thread_ids: list[str] = []
    model_process_started = 0
    raw_event_lines = 0
    hook_allowed = 0
    hook_denied = 0
    invalid_hook_audit: list[dict[str, Any]] = []
    forbidden_source_keys = {'answer', 'ground_truth', 'reference_answer', 'expected_result'}
    forbidden_prompt_fragments = [
        'train_0629.jsonl',
        'data/simulation',
        'saved_configs/',
        'saved_configs\\',
        '10.139.194.154:3080',
    ]

    for attempt_dir in attempt_dirs:
        rel = relative(attempt_dir)
        metadata: dict[str, Any] = {}
        events = attempt_dir / 'events.jsonl'
        stdout = attempt_dir / 'stdout.log'
        metadata_path = attempt_dir / 'metadata.json'
        safe_record_path = attempt_dir / 'source_record.json'
        prompt_path = attempt_dir / 'prompt.txt'
        hook_path = attempt_dir / 'hook_audit.jsonl'
        if not events.exists():
            missing_events.append(rel)
        else:
            with events.open('r', encoding='utf-8', errors='strict') as handle:
                for line_number, line in enumerate(handle, start=1):
                    raw_event_lines += 1
                    try:
                        json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        invalid_events.append({'attempt_path': rel, 'line': line_number})
                        break
        if stdout.exists() and events.exists() and stdout.read_bytes() != events.read_bytes():
            stdout_event_mismatches.append(rel)
        if not metadata_path.exists():
            missing_metadata.append(rel)
        else:
            metadata = load(metadata_path)
            if metadata.get('model_process_started'):
                model_process_started += 1
                if metadata.get('model') != MODEL:
                    wrong_models.append(rel)
                if not metadata.get('ephemeral_session'):
                    non_ephemeral_calls.append(rel)
                thread_id = metadata.get('thread_id')
                if thread_id:
                    thread_ids.append(str(thread_id))
            workdir = metadata.get('working_directory')
            if workdir:
                resolved = Path(workdir).resolve()
                if attempt_dir.resolve() not in (resolved, *resolved.parents):
                    unsafe_workdirs.append(rel)
        if safe_record_path.exists():
            safe_record = load(safe_record_path)
            if forbidden_source_keys.intersection(safe_record):
                unsafe_source_records.append(rel)
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding='utf-8', errors='replace').lower()
            if any(fragment.lower() in prompt for fragment in forbidden_prompt_fragments):
                unsafe_prompts.append(rel)
        if hook_path.exists():
            for line_number, line in enumerate(
                hook_path.read_text(encoding='utf-8', errors='strict').splitlines(),
                start=1,
            ):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_hook_audit.append({
                        'attempt_path': rel,
                        'line': line_number,
                        'expected_infrastructure_failure': (
                            metadata.get('status') == 'infrastructure_failure'
                            and metadata.get('error_class') == 'hook_audit_invalid'
                        ),
                    })
                    continue
                if item.get('allowed'):
                    hook_allowed += 1
                    command = str(item.get('command') or '').lower()
                    if '127.0.0.1:3080' not in command or any(
                        fragment.lower() in command for fragment in forbidden_prompt_fragments
                    ):
                        allowed_hook_violations.append(rel)
                else:
                    hook_denied += 1

    accepted_paths: list[str] = []
    accepted_events: list[str] = []
    accepted_errors: list[dict[str, Any]] = []
    accepted_comparators: Counter[str] = Counter()
    independent_recheck_failures: list[str] = []
    index_errors: list[str] = []
    state_by_row = {sample['row_index']: sample for sample in state['samples']}
    for indexed_sample in accepted_index['samples']:
        row_index = indexed_sample['row_index']
        mapping = indexed_sample['mapping']
        expected_keys = [f'success_{index:02d}' for index in range(1, indexed_sample['accepted_count'] + 1)]
        if list(mapping) != expected_keys:
            index_errors.append(f'row_{row_index}_slots')
        state_sample = state_by_row[row_index]
        if indexed_sample['accepted_count'] != state_sample['accepted_count']:
            index_errors.append(f'row_{row_index}_count')
        for success_key, item in mapping.items():
            attempt_path = EXPERIMENT / item['attempt_path']
            events_path = EXPERIMENT / item['events_path']
            accepted_paths.append(item['attempt_path'])
            accepted_events.append(item['events_path'])
            judgment_path = attempt_path / 'judgment.json'
            metadata_path = attempt_path / 'metadata.json'
            final_path = attempt_path / 'final_answer.txt'
            if not attempt_path.is_dir() or not events_path.is_file() or not judgment_path.is_file():
                accepted_errors.append({'row_index': row_index, 'success': success_key, 'error': 'missing_artifact'})
                continue
            judgment = load(judgment_path)
            metadata = load(metadata_path)
            accepted_comparators[str(judgment.get('comparator'))] += 1
            if not judgment.get('correct') or not judgment.get('parsed'):
                accepted_errors.append({'row_index': row_index, 'success': success_key, 'error': 'judgment_not_correct'})
            if metadata.get('model') != MODEL or metadata.get('generation_status') != 'completed':
                accepted_errors.append({'row_index': row_index, 'success': success_key, 'error': 'metadata_invalid'})
            if item.get('attempt_index') != metadata.get('attempt_index'):
                accepted_errors.append({'row_index': row_index, 'success': success_key, 'error': 'attempt_index_mismatch'})
            prediction = parse_result(final_path.read_text(encoding='utf-8', errors='replace'))
            if not independent_correct(prediction, records[row_index]['answer']):
                independent_recheck_failures.append(item['attempt_path'])

    final_state_errors: list[str] = []
    for sample in state['samples']:
        if len(sample['accepted_attempts']) != sample['accepted_count']:
            final_state_errors.append(f'row_{sample[row_index]}_accepted_count')
        if sample['status'] == 'completed_with_10_correct' and sample['accepted_count'] != 10:
            final_state_errors.append(f'row_{sample[row_index]}_completed_count')
        if sample['status'] == 'abandoned_after_10_consecutive_wrong' and sample['consecutive_wrong'] != 10:
            final_state_errors.append(f'row_{sample[row_index]}_abandoned_count')
        if sample['status'] not in {
            'completed_with_10_correct',
            'abandoned_after_10_consecutive_wrong',
            'stopped_by_infrastructure_blocker',
        }:
            final_state_errors.append(f'row_{sample[row_index]}_status')

    sensitive_patterns = {
        'openai_style_key': re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),
        'bearer_token': re.compile(rb'Bearer\s+[A-Za-z0-9._-]{20,}', re.IGNORECASE),
        'api_key_assignment': re.compile(rb'(?:OPENAI_API_KEY|AZURE_OPENAI_API_KEY)\s*[=:]\s*[^\s]+', re.IGNORECASE),
    }
    sensitive_hits: list[dict[str, str]] = []
    for path in EXPERIMENT.rglob('*'):
        if not path.is_file() or 'runtime' in path.parts:
            continue
        data = path.read_bytes()
        for label, pattern in sensitive_patterns.items():
            if pattern.search(data):
                sensitive_hits.append({'path': relative(path), 'pattern': label})

    status_counts = Counter(sample['status'] for sample in state['samples'])
    source_hash = sha256(DATASET)
    unexpected_invalid_hook_audit = [
        item for item in invalid_hook_audit
        if not item['expected_infrastructure_failure']
    ]
    checks = {
        'source_record_count_100': len(source_lines) == 100,
        'source_hash_unchanged': source_hash == EXPECTED_SOURCE_SHA256,
        'scheduled_sample_count_100': len(state['samples']) == 100,
        'all_samples_terminal': not final_state_errors,
        'accepted_count_matches_index': len(accepted_paths) == sum(sample['accepted_count'] for sample in state['samples']),
        'accepted_attempt_paths_unique': len(accepted_paths) == len(set(accepted_paths)),
        'accepted_event_paths_unique': len(accepted_events) == len(set(accepted_events)),
        'accepted_artifacts_and_judgments_valid': not accepted_errors,
        'accepted_independent_answer_recheck_passed': not independent_recheck_failures,
        'accepted_index_sequential_and_state_consistent': not index_errors,
        'all_attempt_directories_have_events': not missing_events,
        'all_nonempty_event_lines_are_json': not invalid_events,
        'events_are_raw_stdout_streams': not stdout_event_mismatches,
        'all_attempts_have_metadata': not missing_metadata,
        'all_started_model_calls_use_exact_model': not wrong_models,
        'all_started_model_calls_are_ephemeral': not non_ephemeral_calls,
        'all_model_thread_ids_unique': len(thread_ids) == len(set(thread_ids)),
        'attempt_workdirs_are_isolated': not unsafe_workdirs,
        'safe_source_records_exclude_answers': not unsafe_source_records,
        'prompts_exclude_source_and_remote_paths': not unsafe_prompts,
        'allowed_hook_calls_are_local_api_only': not allowed_hook_violations,
        'hook_audit_invalid_lines_confined_to_recorded_infrastructure_failures': not unexpected_invalid_hook_audit,
        'no_sensitive_credential_patterns_found': not sensitive_hits,
        'runner_protected_tree_integrity_passed': bool(summary['integrity']['passed']),
    }
    audit = {
        'schema_version': 'ip-distill-final-audit.v1',
        'experiment_root': str(EXPERIMENT),
        'source_path': str(DATASET),
        'source_sha256': source_hash,
        'source_record_count': len(source_lines),
        'sample_count': len(state['samples']),
        'status_counts': dict(sorted(status_counts.items())),
        'accepted_total': len(accepted_paths),
        'attempt_directory_count': len(attempt_dirs),
        'model_process_started_count': model_process_started,
        'raw_event_line_count': raw_event_lines,
        'thread_id_count': len(thread_ids),
        'unique_thread_id_count': len(set(thread_ids)),
        'accepted_comparator_counts': dict(sorted(accepted_comparators.items())),
        'hook_allowed_count': hook_allowed,
        'hook_denied_count': hook_denied,
        'checks': checks,
        'failures': {
            'missing_events': missing_events,
            'invalid_events': invalid_events,
            'stdout_event_mismatches': stdout_event_mismatches,
            'missing_metadata': missing_metadata,
            'wrong_models': wrong_models,
            'non_ephemeral_calls': non_ephemeral_calls,
            'duplicate_thread_ids': [item for item, count in Counter(thread_ids).items() if count > 1],
            'unsafe_workdirs': unsafe_workdirs,
            'unsafe_source_records': unsafe_source_records,
            'unsafe_prompts': unsafe_prompts,
            'allowed_hook_violations': sorted(set(allowed_hook_violations)),
            'invalid_hook_audit': invalid_hook_audit,
            'unexpected_invalid_hook_audit': unexpected_invalid_hook_audit,
            'accepted_errors': accepted_errors,
            'independent_recheck_failures': independent_recheck_failures,
            'index_errors': index_errors,
            'final_state_errors': final_state_errors,
            'sensitive_hits': sensitive_hits,
        },
        'passed': all(checks.values()),
    }
    atomic_json(REPORT / 'final_audit.json', audit)
    print(json.dumps({
        'passed': audit['passed'],
        'accepted_total': audit['accepted_total'],
        'attempt_directory_count': audit['attempt_directory_count'],
        'model_process_started_count': audit['model_process_started_count'],
        'raw_event_line_count': audit['raw_event_line_count'],
        'failed_checks': [key for key, value in checks.items() if not value],
    }, ensure_ascii=False))
    return 0 if audit['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
