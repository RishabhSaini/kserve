# Copyright 2025 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import os
import time

import pytest
import requests
from kserve import KServeClient
from kubernetes import client

from .diagnostic import collect_pod_logs, print_all_events_table
from .fixtures import (
    KSERVE_TEST_NAMESPACE,
    create_inference_objectives,
    delete_inference_objectives,
    inject_k8s_proxy,
)
from .logging import log_execution, logger
from .test_llm_inference_service import (
    TestCase,
    completions_payload,
    create_response_assertion,
    create_llmisvc,
    get_llmisvc,
    maybe_delete_llmisvc,
    wait_for_llm_isvc_ready,
    wait_for_model_response,
)

DROPPED_REASON_HEADER = "x-llm-d-request-dropped-reason"

FC_OBJECTIVES = [
    {"name": "fc-high-priority", "priority": 100},
    {"name": "fc-medium-priority", "priority": 50},
    {"name": "fc-low-priority", "priority": -1},
]
FC_OBJECTIVE_NAMES = [o["name"] for o in FC_OBJECTIVES]


@pytest.mark.llminferenceservice
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            TestCase(
                base_refs=[
                    "router-managed",
                    "scheduler-flow-control-round-robin",
                    "workload-llmd-simulator",
                ],
                prompt="KServe is a",
                service_name="fc-smoke-test",
                payload_formatter=completions_payload,
                response_assertion=create_response_assertion(with_field="choices"),
            ),
            marks=[
                pytest.mark.cluster_cpu,
                pytest.mark.cluster_single_node,
                pytest.mark.llmd_simulator,
                pytest.mark.flow_control,
            ],
            id="flow-control-smoke",
        ),
    ],
    indirect=True,
)
def test_flow_control_smoke(test_case: TestCase):
    """Verify that the EPP boots and serves traffic with flow control enabled.

    Uses the most complex flow control config (round-robin fairness, flowControl gate,
    utilization-detector) to cover config wiring for all policy types in one test.
    """
    inject_k8s_proxy()
    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config"),
        client_configuration=client.Configuration(),
    )

    service_name = test_case.llm_service.metadata.name
    prefix = test_case.log_prefix
    test_failed = False
    try:
        print(f"{prefix} Creating LLMInferenceService {service_name}")
        create_llmisvc(kserve_client, test_case.llm_service)
        print(f"{prefix} Waiting for ready")
        wait_for_llm_isvc_ready(
            kserve_client, test_case.llm_service, test_case.wait_timeout
        )
        print(f"{prefix} Waiting for model response")
        wait_for_model_response(kserve_client, test_case, test_case.wait_timeout)
    except Exception as e:
        test_failed = True
        logger.error(f"{prefix} Failed: {e}")
        _collect_diagnostics(kserve_client, test_case.llm_service)
        raise
    finally:
        maybe_delete_llmisvc(kserve_client, test_case.llm_service, test_failed)


@pytest.mark.llminferenceservice
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            TestCase(
                base_refs=[
                    "router-managed",
                    "scheduler-flow-control-default",
                    "workload-llmd-simulator",
                ],
                prompt="KServe is a",
                service_name="fc-priorities-test",
                payload_formatter=completions_payload,
                response_assertion=create_response_assertion(with_field="choices"),
            ),
            marks=[
                pytest.mark.cluster_cpu,
                pytest.mark.cluster_single_node,
                pytest.mark.llmd_simulator,
                pytest.mark.flow_control,
            ],
            id="priority-bands",
        ),
    ],
    indirect=True,
)
def test_flow_control_priority_bands(test_case: TestCase):
    """Verify that InferenceObjectives provision priority bands and requests at each priority succeed.

    This exercises the lifecycle interaction between the control plane (InferenceObjective reconciler)
    and the flow control data plane (band provisioning, flow creation, dispatch). This interaction
    has had real production bugs (issue #1606).

    After creating objectives, verifies the EPP logs confirm band provisioning before sending traffic.
    """
    inject_k8s_proxy()
    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config"),
        client_configuration=client.Configuration(),
    )

    service_name = test_case.llm_service.metadata.name
    prefix = test_case.log_prefix
    test_failed = False
    try:
        create_llmisvc(kserve_client, test_case.llm_service)
        wait_for_llm_isvc_ready(
            kserve_client, test_case.llm_service, test_case.wait_timeout
        )
        wait_for_model_response(kserve_client, test_case, test_case.wait_timeout)

        pool_name = f"{service_name}-inference-pool"
        create_inference_objectives(pool_name, FC_OBJECTIVES)
        time.sleep(5)

        # Verify bands were actually provisioned by checking EPP logs.
        epp_logs = _get_epp_logs(service_name, since_seconds=30)
        for obj in FC_OBJECTIVES:
            expected_priority = obj["priority"]
            assert any(
                "Provisioning priority band" in line and str(expected_priority) in line
                for line in epp_logs
            ), (
                f"EPP logs do not confirm provisioning of priority band {expected_priority}. "
                f"The request may succeed via priority-0 fallback, masking a provisioning failure."
            )
            print(f"{prefix} Confirmed band provisioned for priority={expected_priority}")

        url = _get_service_url(kserve_client, test_case)
        payload = test_case.payload_formatter(test_case)

        for obj_name, priority in [
            ("fc-high-priority", 100),
            ("fc-medium-priority", 50),
            ("fc-low-priority", -1),
        ]:
            print(f"{prefix} Sending request with objective={obj_name} (priority={priority})")
            resp = requests.post(
                f"{url}/v1/completions",
                json=payload,
                headers={"x-gateway-inference-objective": obj_name},
                timeout=test_case.response_timeout,
            )
            assert resp.status_code == 200, (
                f"Request at priority {priority} failed: "
                f"{resp.status_code} {resp.text}"
            )

        print(f"{prefix} Sending request with default priority (no objective)")
        resp = requests.post(
            f"{url}/v1/completions", json=payload, timeout=test_case.response_timeout
        )
        assert resp.status_code == 200, (
            f"Default priority request failed: {resp.status_code} {resp.text}"
        )
    except Exception as e:
        test_failed = True
        logger.error(f"{prefix} Failed: {e}")
        _collect_diagnostics(kserve_client, test_case.llm_service)
        raise
    finally:
        delete_inference_objectives(FC_OBJECTIVE_NAMES)
        maybe_delete_llmisvc(kserve_client, test_case.llm_service, test_failed)


@pytest.mark.llminferenceservice
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            TestCase(
                base_refs=[
                    "router-managed",
                    "scheduler-flow-control-concurrency-detector",
                    "workload-llmd-simulator-slow",
                ],
                prompt="Write a long essay about computing",
                service_name="fc-saturation-test",
                max_tokens=100,
                payload_formatter=completions_payload,
                wait_timeout=900,
                response_timeout=30,
            ),
            marks=[
                pytest.mark.cluster_cpu,
                pytest.mark.cluster_single_node,
                pytest.mark.llmd_simulator,
                pytest.mark.flow_control,
            ],
            id="concurrency-saturation",
        ),
    ],
    indirect=True,
)
def test_flow_control_saturation(test_case: TestCase):
    """Verify that flow control rejects requests under saturation and recovers afterward.

    Uses a slow simulator (max-num-seqs=1, 500ms latency) with a concurrency detector
    (maxConcurrency=2) and TTL=5s. Sends concurrent requests and verifies:
    1. At least one request succeeds (dispatch works).
    2. Rejected requests get 429 or 503 with a dropped-reason header (not 500).
    3. After saturation subsides, a follow-up request succeeds (system recovers).
    """
    inject_k8s_proxy()
    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config"),
        client_configuration=client.Configuration(),
    )

    service_name = test_case.llm_service.metadata.name
    prefix = test_case.log_prefix
    test_failed = False
    try:
        create_llmisvc(kserve_client, test_case.llm_service)
        wait_for_llm_isvc_ready(
            kserve_client, test_case.llm_service, test_case.wait_timeout
        )
        wait_for_model_response(kserve_client, test_case, test_case.wait_timeout)

        url = _get_service_url(kserve_client, test_case)
        payload = test_case.payload_formatter(test_case)

        # --- Phase 1: Saturate ---
        num_requests = 6
        print(
            f"{prefix} Sending {num_requests} concurrent requests "
            f"(backend max-num-seqs=1, concurrency limit=2, TTL=5s)"
        )

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as pool:
            futures = [
                pool.submit(_send_request_with_headers, url, "/v1/completions", payload, test_case.response_timeout)
                for _ in range(num_requests)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        statuses = [r["status"] for r in results]
        successful = [r for r in results if r["status"] == 200]
        rejected = [r for r in results if r["status"] in (429, 503)]
        other = [r for r in results if r["status"] not in (200, 429, 503)]

        print(
            f"{prefix} Results: {len(successful)} succeeded, {len(rejected)} rejected/expired, "
            f"{len(other)} other. Codes: {sorted(statuses)}"
        )

        assert len(successful) > 0, "At least one request must succeed"
        assert not other, f"Unexpected status codes (expected 200/429/503): {[r['status'] for r in other]}"

        # Verify rejected responses carry the dropped-reason header.
        for r in rejected:
            reason = r["headers"].get(DROPPED_REASON_HEADER, "")
            print(f"{prefix} Rejected {r['status']}: {DROPPED_REASON_HEADER}={reason}")
            assert reason, (
                f"Rejected response (HTTP {r['status']}) is missing the "
                f"{DROPPED_REASON_HEADER} header"
            )

        # --- Phase 2: Recovery ---
        print(f"{prefix} Waiting for saturation to subside...")
        time.sleep(10)

        print(f"{prefix} Sending follow-up request to verify recovery")
        recovery_resp = requests.post(
            f"{url}/v1/completions", json=payload, timeout=test_case.response_timeout
        )
        assert recovery_resp.status_code == 200, (
            f"Recovery request failed after saturation subsided: "
            f"{recovery_resp.status_code} {recovery_resp.text}"
        )
        print(f"{prefix} Recovery confirmed: HTTP {recovery_resp.status_code}")

    except Exception as e:
        test_failed = True
        logger.error(f"{prefix} Failed: {e}")
        _collect_diagnostics(kserve_client, test_case.llm_service)
        raise
    finally:
        maybe_delete_llmisvc(kserve_client, test_case.llm_service, test_failed)


# --- Helpers ---


def _send_request_with_headers(base_url, endpoint, payload, timeout):
    try:
        resp = requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout)
        return {
            "status": resp.status_code,
            "headers": dict(resp.headers),
        }
    except requests.exceptions.Timeout:
        return {"status": 408, "headers": {}}
    except Exception:
        return {"status": 0, "headers": {}}


def _get_service_url(kserve_client, test_case):
    llm_isvc = test_case.llm_service
    status = get_llmisvc(
        kserve_client,
        llm_isvc.metadata.name,
        llm_isvc.metadata.namespace,
        llm_isvc.api_version.split("/")[1],
    )
    addresses = status.get("status", {}).get("addresses", [])
    assert addresses, f"No addresses in status: {status.get('status')}"
    return addresses[0].get("url", "").rstrip("/")


def _get_epp_logs(service_name, since_seconds=60):
    core_v1 = client.CoreV1Api()
    pods = core_v1.list_namespaced_pod(
        namespace=KSERVE_TEST_NAMESPACE,
        label_selector=f"serving.kserve.io/llminferenceservice={service_name}",
    )
    lines = []
    for pod in pods.items:
        for container in pod.spec.containers:
            if "epp" not in container.name and "scheduler" not in container.name:
                continue
            try:
                log = core_v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=KSERVE_TEST_NAMESPACE,
                    container=container.name,
                    since_seconds=since_seconds,
                )
                lines.extend(log.splitlines())
            except client.rest.ApiException:
                pass
    return lines


def _collect_diagnostics(kserve_client, llm_isvc):
    try:
        print_all_events_table(llm_isvc.metadata.namespace)
        collect_pod_logs(
            llm_isvc.metadata.namespace,
            label_selector=f"serving.kserve.io/llminferenceservice={llm_isvc.metadata.name}",
        )
    except Exception as diag_err:
        logger.warning(f"Failed to collect diagnostics: {diag_err}")
