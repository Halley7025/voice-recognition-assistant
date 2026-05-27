"""Verification gate: intercept protected commands and require voiceprint verification."""
from logger_config import setup_logger
_log = setup_logger(__name__)

# Commands that require voiceprint verification before execution
PROTECTED_COMMANDS = {
    # App management
    "open_app", "close_app",
    # System power
    "shutdown", "restart", "logout", "sleep",
    # Destructive operations
    "empty_recycle", "delete_file", "clean_temp",
    # System settings
    "open_settings", "open_task_manager",
}


def is_protected(cmd):
    """Check if a command (including param variants) requires verification."""
    if not cmd:
        return False
    # Handle "intent:param" format
    base_cmd = cmd.split(":")[0] if ":" in cmd else cmd
    return base_cmd in PROTECTED_COMMANDS


def verify_and_execute(cmd, raw_audio, verifier, controller, preprocessor):
    """Execute a command with voiceprint verification gate.

    Args:
        cmd: The parsed command string (e.g., "open_app:网易云音乐").
        raw_audio: The raw audio numpy array from the microphone.
        verifier: SpeakerVerifier instance.
        controller: SystemController instance.
        preprocessor: AudioPreprocessor instance.

    Returns:
        (success, result, verified, user_id, similarity)
    """
    if not is_protected(cmd):
        # Not a protected command - execute directly
        success, result = controller.run(cmd)
        return success, result, None, None, None

    # Protected command - verify speaker first
    if verifier is None or verifier.model is None:
        _log.warning("声纹模型未加载，跳过验证")
        success, result = controller.run(cmd)
        return success, result, None, None, None

    if not verifier.list_users():
        _log.warning("无已注册用户，跳过验证")
        success, result = controller.run(cmd)
        return success, result, None, None, None

    # Process audio for speaker verification
    processed = preprocessor.process_for_speaker(raw_audio)
    is_verified, user_id, similarity = verifier.verify_any_user(processed)

    if is_verified:
        _log.info(f"验证通过({user_id}, sim={similarity:.3f}), 执行: {cmd}")
        success, result = controller.run(cmd)
        return success, result, True, user_id, similarity
    else:
        msg = f"声纹验证失败 (相似度: {similarity:.3f}), 指令被拒绝"
        _log.warning(msg)
        return False, msg, False, user_id, similarity
