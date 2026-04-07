"""
Hardware and software environment information collection.

Collects GPU, CPU, OS, and software version information for
benchmark reproducibility.
"""

import os
import platform
import subprocess
import re
from typing import Dict, Any, Optional

from ..models import HardwareInfo
from ..config import config


def run_command(cmd: str, default: str = "") -> str:
    """
    Run a shell command and return output.

    Args:
        cmd: Command to run
        default: Default value if command fails

    Returns:
        Command output or default
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else default
    except Exception:
        return default


def _get_nvidia_gpu_info(info: Dict[str, Any]) -> bool:
    """
    Populate *info* from nvidia-smi.  Returns True if NVIDIA GPU was found.
    """
    if not run_command("which nvidia-smi"):
        return False

    name = run_command("nvidia-smi --query-gpu=name --format=csv,noheader,nounits")
    if name:
        info["model"] = name.split('\n')[0].strip()

    vram = run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    if vram:
        try:
            info["vram_gb"] = float(vram.split('\n')[0].strip()) / 1024
        except ValueError:
            pass

    driver = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
    if driver:
        info["driver_version"] = driver.split('\n')[0].strip()

    power = run_command("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits")
    if power:
        try:
            info["power_draw_watts"] = float(power.split('\n')[0].strip())
        except ValueError:
            pass

    return bool(info["model"])


def _get_rocm_gpu_info(info: Dict[str, Any]) -> bool:
    """
    Populate *info* from rocm-smi (AMD ROCm).  Returns True if AMD GPU was found.
    """
    if not run_command("which rocm-smi"):
        return False

    # rocm-smi --showproductname prints lines like "GPU[0] : Product Name: Radeon RX 7900 XTX"
    name = run_command("rocm-smi --showproductname --noheader 2>/dev/null | awk -F': ' '/Product Name/{print $NF; exit}'")
    if name:
        info["model"] = name.strip()

    # VRAM in bytes → GB
    vram = run_command("rocm-smi --showmeminfo vram --noheader 2>/dev/null | awk '/Total Memory/{print $NF; exit}'")
    if vram:
        try:
            info["vram_gb"] = float(vram.strip()) / (1024 ** 3)
        except ValueError:
            pass

    driver = run_command("rocm-smi --showdriverversion --noheader 2>/dev/null | awk '{print $NF; exit}'")
    if driver:
        info["driver_version"] = driver.strip()

    return bool(info["model"])


def _get_metal_gpu_info(info: Dict[str, Any]) -> bool:
    """
    Populate *info* from system_profiler (Apple Metal / macOS).
    Returns True if a Metal GPU was found.
    """
    if platform.system() != "Darwin":
        return False

    sp_out = run_command("system_profiler SPDisplaysDataType 2>/dev/null")
    if not sp_out:
        return False

    match = re.search(r'Chipset Model:\s*(.+)', sp_out)
    if match:
        info["model"] = match.group(1).strip()

    match = re.search(r'VRAM \([^)]+\):\s*([\d.]+)\s*(MB|GB)', sp_out, re.IGNORECASE)
    if match:
        try:
            vram_val = float(match.group(1))
            if match.group(2).upper() == "MB":
                vram_val /= 1024
            info["vram_gb"] = vram_val
        except ValueError:
            pass

    return bool(info["model"])


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information, trying NVIDIA, AMD ROCm, and Apple Metal in order.

    Returns:
        Dictionary with GPU model, VRAM, driver version, and power draw
    """
    info = {
        "model": "",
        "vram_gb": 0.0,
        "driver_version": "",
        "power_draw_watts": 0.0
    }

    _get_nvidia_gpu_info(info) or _get_rocm_gpu_info(info) or _get_metal_gpu_info(info)
    return info


def get_cuda_version() -> str:
    """
    Get the GPU accelerator version (CUDA, ROCm, or Metal).

    Returns:
        Version string for the active GPU accelerator, or empty string.
    """
    # CUDA — try nvcc then nvidia-smi
    nvcc_version = run_command("nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//'")
    if nvcc_version:
        return nvcc_version

    nvidia_smi_output = run_command("nvidia-smi")
    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', nvidia_smi_output)
    if match:
        return match.group(1)

    # ROCm — hipconfig
    rocm_version = run_command("hipconfig --version 2>/dev/null | head -1")
    if rocm_version:
        return rocm_version

    # Metal — macOS build version (Metal is always present on modern macOS)
    if platform.system() == "Darwin":
        macos_ver = run_command("sw_vers -productVersion 2>/dev/null")
        if macos_ver:
            return f"Metal (macOS {macos_ver.strip()})"

    return ""


def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU information.

    Returns:
        Dictionary with CPU model and core count
    """
    info = {
        "model": platform.processor() or "",
        "cores": os.cpu_count() or 0
    }

    # Try to get more detailed CPU info on Linux
    if platform.system() == "Linux":
        model = run_command("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2")
        if model:
            info["model"] = model.strip()

    return info


def get_memory_info() -> float:
    """
    Get total system RAM in GB.

    Returns:
        RAM in GB
    """
    if platform.system() == "Linux":
        mem_kb = run_command("cat /proc/meminfo | grep MemTotal | awk '{print $2}'")
        if mem_kb:
            try:
                return float(mem_kb) / (1024 * 1024)
            except ValueError:
                pass

    # Fallback - try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    return 0.0


def get_os_info() -> Dict[str, str]:
    """
    Get OS and kernel information.

    Returns:
        Dictionary with OS name and kernel version
    """
    return {
        "os_name": f"{platform.system()} {platform.release()}",
        "kernel_version": platform.version()
    }


def get_k3s_version() -> str:
    """
    Get K3s version.

    Returns:
        K3s version string
    """
    version = run_command("k3s --version | head -1")
    if version:
        match = re.search(r'v[\d.]+', version)
        if match:
            return match.group(0)
    return ""


def get_llama_cpp_version() -> str:
    """
    Get llama.cpp version from running server.

    Returns:
        llama.cpp version or commit hash
    """
    # Try to get from llama-server
    # This would require querying the running server
    # For now, return empty - can be populated from server response
    return ""


def get_model_info() -> Dict[str, str]:
    """
    Get model information from config.

    Returns:
        Dictionary with model name and quantization
    """
    model_name = config.model_name
    quantization = ""

    # Extract quantization from model filename
    # e.g., "Qwen3-14B-Q4_K_M.gguf" -> "Q4_K_M"
    match = re.search(r'(Q\d+_K(?:_[A-Z])?)', model_name, re.IGNORECASE)
    if match:
        quantization = match.group(1).upper()

    return {
        "name": model_name,
        "quantization": quantization
    }


def collect_hardware_info() -> HardwareInfo:
    """
    Collect all hardware and software information.

    Returns:
        HardwareInfo dataclass with all collected data
    """
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    os_info = get_os_info()
    model_info = get_model_info()

    return HardwareInfo(
        gpu_model=gpu_info["model"],
        gpu_vram_gb=gpu_info["vram_gb"],
        gpu_driver_version=gpu_info["driver_version"],
        cuda_version=get_cuda_version(),
        cpu_model=cpu_info["model"],
        cpu_cores=cpu_info["cores"],
        ram_gb=get_memory_info(),
        os_name=os_info["os_name"],
        kernel_version=os_info["kernel_version"],
        k3s_version=get_k3s_version(),
        llama_cpp_version=get_llama_cpp_version(),
        model_name=model_info["name"],
        model_quantization=model_info["quantization"],
        context_length=0,  # Can be populated from server query
        power_draw_watts=gpu_info["power_draw_watts"]
    )


def hardware_info_to_markdown(info: HardwareInfo) -> str:
    """
    Generate Markdown table of hardware information.

    Args:
        info: HardwareInfo to format

    Returns:
        Formatted Markdown string
    """
    lines = [
        "## Hardware Information",
        "",
        "### GPU",
        f"- Model: {info.gpu_model or 'N/A'}",
        f"- VRAM: {info.gpu_vram_gb:.1f} GB" if info.gpu_vram_gb else "- VRAM: N/A",
        f"- Driver: {info.gpu_driver_version or 'N/A'}",
        f"- CUDA: {info.cuda_version or 'N/A'}",
        f"- Power Draw: {info.power_draw_watts:.0f}W" if info.power_draw_watts else "- Power Draw: N/A",
        "",
        "### CPU",
        f"- Model: {info.cpu_model or 'N/A'}",
        f"- Cores: {info.cpu_cores or 'N/A'}",
        "",
        "### System",
        f"- RAM: {info.ram_gb:.1f} GB" if info.ram_gb else "- RAM: N/A",
        f"- OS: {info.os_name or 'N/A'}",
        f"- Kernel: {info.kernel_version or 'N/A'}",
        "",
        "### Software",
        f"- K3s: {info.k3s_version or 'N/A'}",
        f"- llama.cpp: {info.llama_cpp_version or 'N/A'}",
        "",
        "### Model",
        f"- Name: {info.model_name or 'N/A'}",
        f"- Quantization: {info.model_quantization or 'N/A'}",
        f"- Context Length: {info.context_length or 'N/A'}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Test hardware info collection
    print("Collecting hardware information...")
    info = collect_hardware_info()
    print(hardware_info_to_markdown(info))
    print("\n--- JSON Output ---")
    print(info.to_dict())
