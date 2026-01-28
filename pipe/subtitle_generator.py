import pysubs2
from typing import List, Tuple
from pipe.config import Config

class SubtitleGenerator:
    @staticmethod
    def generate_clean_srt(master_index: List[dict], intervals: List[Tuple[float, float]], output_path: str):
        """
        Generates a standard SRT file for the 'Clean Long-form' video.
        
        Logic:
        1. The 'Clean Video' is a concatenation of specific 'intervals' from the Raw Video.
        2. We must find which words fall into these intervals.
        3. We must shift their timestamps to match the new continuous timeline.
        4. We group the shifted words into readable subtitle lines (max chars/punctuation).
        """
        print(f"--- Generating YouTube CC (SRT) ---")
        
        # 1. Collect and Shift Words
        shifted_words = [] # List of {'word': str, 'start': float, 'end': float}
        current_clean_cursor = 0.0
        
        # Optimization: Keep track of where we are in the master_index to avoid O(N^2)
        idx_ptr = 0
        total_segs = len(master_index)

        # Sort intervals just in case
        intervals = sorted(intervals, key=lambda x: x[0])

        for (chunk_start, chunk_end) in intervals:
            chunk_duration = chunk_end - chunk_start
            
            # Advance ptr to the first segment that could possibly start after chunk_start
            while idx_ptr < total_segs and master_index[idx_ptr]['end'] < chunk_start:
                idx_ptr += 1
            
            # Scan segments that overlap with this chunk
            temp_ptr = idx_ptr
            while temp_ptr < total_segs:
                seg = master_index[temp_ptr]
                
                # If segment starts after this chunk ends, we can stop scanning for this chunk
                if seg['start'] > chunk_end:
                    break
                
                # Check individual words
                if 'words' in seg:
                    for w in seg['words']:
                        # Strict inclusion: Word must be fully inside the kept audio chunk
                        if w['start'] >= chunk_start and w['end'] <= chunk_end:
                            offset = w['start'] - chunk_start
                            shifted_words.append({
                                'word': w['word'],
                                'start': current_clean_cursor + offset,
                                'end': current_clean_cursor + offset + (w['end'] - w['start'])
                            })
                
                temp_ptr += 1
            
            # Advance the cursor for the next chunk
            current_clean_cursor += chunk_duration

        # 2. Group into Subtitles (SRT Formatting)
        subs = pysubs2.SSAFile()
        buffer = []
        MAX_CHARS = 42
        MAX_DURATION_GAP = 1.0 # seconds

        for i, w in enumerate(shifted_words):
            buffer.append(w)
            
            # Determine if we should flush the buffer to a subtitle line
            current_text = " ".join([b['word'] for b in buffer]).strip()
            
            should_flush = False
            
            # Condition A: Punctuation (Natural break)
            if w['word'].strip()[-1] in ".?!":
                should_flush = True
            
            # Condition B: Character limit
            elif len(current_text) > MAX_CHARS:
                should_flush = True
                
            # Condition C: Large gap to next word
            elif i < len(shifted_words) - 1:
                next_start = shifted_words[i+1]['start']
                if next_start - w['end'] > MAX_DURATION_GAP:
                    should_flush = True
            
            # Condition D: End of list
            elif i == len(shifted_words) - 1:
                should_flush = True

            if should_flush:
                start_ms = int(buffer[0]['start'] * 1000)
                end_ms = int(buffer[-1]['end'] * 1000)
                
                # Minimum duration constraint (human readability)
                if end_ms - start_ms < 500: 
                    end_ms = start_ms + 500
                    
                subs.events.append(pysubs2.SSAEvent(
                    start=start_ms, 
                    end=end_ms, 
                    text=current_text
                ))
                buffer = []

        # 3. Save
        subs.save(str(output_path), format="srt")
        print(f"-> SRT saved to {output_path}")